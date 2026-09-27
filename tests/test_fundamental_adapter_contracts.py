"""Exercise AkShare argument and stock-scope contracts through the real adapter."""

from datetime import datetime
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _recent_report_dates,
)


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 1, 1), ["20251231", "20250930"]),
    (datetime(2024, 3, 31), ["20231231", "20230930"]),
    (datetime(2024, 4, 1), ["20240331", "20231231"]),
    (datetime(2026, 9, 27), ["20260630", "20260331"]),
])
def test_recent_report_dates_follow_completed_quarters(now, expected):
    assert _recent_report_dates(now) == expected


def test_bulk_endpoints_filter_target_before_stopping_period_fallback(monkeypatch):
    calls = []

    def forecast(date):
        calls.append(("forecast", date))
        code = "000001" if date == "20260630" else "600519"
        return pd.DataFrame({"股票代码": [code], "预告": ["目标预告"]})

    def quick(date):
        calls.append(("quick", date))
        # A nonempty market table without a code cannot establish stock identity.
        if date == "20260630":
            return pd.DataFrame({"快报": ["未知股票"]})
        return pd.DataFrame({"股票代码": ["600519"], "快报": ["目标快报"]})

    def institution(symbol):
        calls.append(("institution", symbol))
        code = "000001" if symbol == "20262" else "600519"
        return pd.DataFrame({"证券代码": [code], "机构数变化": [3]})

    def top10(symbol, date):
        calls.append(("top10", symbol, date))
        assert symbol == "sh600519"
        return pd.DataFrame({"股东名称": ["某股东"], "增减": [100]})

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_yjyg_em=forecast, stock_yjkb_em=quick,
        stock_institute_hold=institution, stock_gdfx_top_10_em=top10,
    ))
    monkeypatch.setattr(
        "data_provider.fundamental_adapter._recent_report_dates",
        lambda: ["20260630", "20260331"],
    )
    result = AkshareFundamentalAdapter().get_fundamental_bundle("600519.SH")

    assert result["errors"] == ["stock_yjkb_em:ValueError"]
    assert result["earnings"]["forecast_summary"] == "目标预告"
    assert result["earnings"]["quick_report_summary"] == "目标快报"
    assert result["institution"] == {
        "institution_holding_change": 3.0, "top10_holder_change": 100.0,
    }
    assert calls == [
        ("forecast", "20260630"), ("forecast", "20260331"),
        ("quick", "20260630"), ("quick", "20260331"),
        ("institution", "20262"), ("institution", "20261"),
        ("top10", "sh600519", "20260630"),
    ]


@pytest.mark.parametrize("code, expected", [
    ("600519", "sh600519"), ("000001.SZ", "sz000001"),
    ("920002", "bj920002"), ("SH688111", "sh688111"),
])
def test_top10_keeps_stock_scope_and_errors_without_unrelated_fallback(
    monkeypatch, code, expected,
):
    calls = []

    def top10(symbol, date):
        calls.append((symbol, date))
        raise KeyError("sdgd")

    def unrelated(**kwargs):
        pytest.fail("A different indicator or a default stock must not be used")

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_gdfx_top_10_em=top10,
        stock_zh_a_gdhs_detail_em=unrelated,
        stock_institute_recommend=unrelated,
        stock_yjbb_em=unrelated,
    ))
    result = AkshareFundamentalAdapter().get_fundamental_bundle(code)
    assert calls == [(expected, date) for date in _recent_report_dates()]
    assert result["institution"] == {}
    assert result["errors"] == ["stock_gdfx_top_10_em:KeyError"] * 2
    assert result["status"] == "not_supported"


def test_installed_akshare_receives_valid_parameters_at_http_boundary(monkeypatch):
    # Keep the real AkShare functions: mocking the adapter or permissive **kwargs
    # stubs would hide signature errors and upstream request construction.
    import akshare as ak
    import requests

    calls = []

    def stop_at_http(url, **kwargs):
        calls.append((url, dict(kwargs.get("params", {}))))
        raise RuntimeError("offline transport boundary")

    monkeypatch.setattr(requests, "get", stop_at_http)
    for name in (
        "stock_financial_abstract", "stock_financial_analysis_indicator",
        "stock_fhps_detail_em", "stock_history_dividend_detail", "stock_dividend_cninfo",
    ):
        monkeypatch.setattr(ak, name, lambda **kwargs: pd.DataFrame(), raising=False)
    monkeypatch.setattr(
        "data_provider.fundamental_adapter._recent_report_dates",
        lambda: ["20260630", "20260331"],
    )
    result = AkshareFundamentalAdapter().get_fundamental_bundle("000001")

    assert len(calls) == 8  # two bounded periods per endpoint, no no-arg calls
    assert not any("TypeError" in error for error in result["errors"])
    period_filters = [params["filter"] for _, params in calls if "filter" in params]
    assert len(period_filters) == 4
    assert all("2026-06-30" in value or "2026-03-31" in value for value in period_filters)
    shareholder_calls = [params for url, params in calls if "PageSDGD" in url]
    assert shareholder_calls == [
        {"code": "SZ000001", "date": "2026-06-30"},
        {"code": "SZ000001", "date": "2026-03-31"},
    ]
    institution_calls = [params for _, params in calls if "reportdate" in params]
    assert [(params["reportdate"], params["quarter"]) for params in institution_calls] == [
        ("2026", "2"), ("2026", "1"),
    ]
