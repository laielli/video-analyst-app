"""ALLOWED_ORIGINS -> CORS origin-list parsing (production deploy wiring).

The API's CORS allow-list is the dev origins plus whatever ALLOWED_ORIGINS carries.
The deploy workflow sets ALLOWED_ORIGINS to the Static Web App URL so the cross-origin
SSE run stream (EventSource issues cross-origin GETs) isn't blocked in production. This
pins the parsing contract: unset/empty stays dev-only, real origins come through, and
sloppy config (spaces, blank entries, a trailing comma) never yields a bogus "" origin.
"""
from __future__ import annotations

import server


def test_empty_env_yields_no_prod_origins():
    # The pre-deploy default: nothing added on top of the dev origins.
    assert server._parse_prod_origins("") == []


def test_whitespace_only_yields_no_prod_origins():
    assert server._parse_prod_origins("   ") == []


def test_single_origin():
    assert server._parse_prod_origins("https://glassbox-web.azurestaticapps.net") == [
        "https://glassbox-web.azurestaticapps.net"
    ]


def test_multiple_origins_are_split_and_trimmed():
    raw = "https://a.example.net, https://b.example.net ,https://c.example.net"
    assert server._parse_prod_origins(raw) == [
        "https://a.example.net",
        "https://b.example.net",
        "https://c.example.net",
    ]


def test_trailing_comma_and_blank_entries_dropped():
    assert server._parse_prod_origins("https://a.example.net,,  ,") == [
        "https://a.example.net"
    ]


def test_dev_origins_are_always_present():
    # The dev localhost origins must never depend on ALLOWED_ORIGINS being set.
    assert "http://localhost:3000" in server._DEV_ORIGINS
    assert "http://127.0.0.1:3000" in server._DEV_ORIGINS
