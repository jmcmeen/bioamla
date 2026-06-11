"""Tests for bioamla.common.http.

Network is never hit: the requests.Session on each APIClient is replaced with a
Mock that returns fake responses.
"""

from unittest.mock import MagicMock

import pytest

from bioamla.common import http as http_mod
from bioamla.common.http import (
    APICache,
    APIClient,
    ConfigAwareMixin,
    RateLimiter,
    clear_cache,
    config_aware,
    config_aware_class,
    get_cache_dir,
    rate_limited,
)


@pytest.fixture(autouse=True)
def isolate_cache_dir(tmp_path, monkeypatch):
    """Point the module cache dir at tmp so we never touch ~/.cache."""
    monkeypatch.setattr(http_mod, "_CACHE_DIR", tmp_path / "apicache")
    yield


def make_response(*, json_data=None, status_ok=True, content_chunks=None, headers=None):
    """Build a fake requests.Response-like Mock."""
    resp = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if not status_ok:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError("boom")
    else:
        resp.raise_for_status.return_value = None
    resp.headers = headers or {}
    if content_chunks is not None:
        resp.iter_content.return_value = content_chunks
    return resp


class TestCacheDir:
    def test_get_cache_dir_creates(self, tmp_path):
        d = get_cache_dir()
        assert d.exists()

    def test_clear_cache(self, tmp_path):
        d = get_cache_dir()
        (d / "a.json").write_text("{}")
        (d / "b.json").write_text("{}")
        assert clear_cache() == 2
        assert list(d.glob("*.json")) == []


class TestAPICache:
    def test_make_key_stable(self):
        c = APICache()
        k1 = c.make_key("http://x", {"a": 1, "b": 2})
        k2 = c.make_key("http://x", {"b": 2, "a": 1})
        assert k1 == k2

    def test_set_get_round_trip(self):
        c = APICache()
        key = c.make_key("http://x", {"q": "t"})
        c.set(key, {"result": 1})
        assert c.get(key) == {"result": 1}

    def test_get_miss_returns_none(self):
        c = APICache()
        assert c.get("missingkey") is None

    def test_disabled_get_set(self):
        c = APICache(enabled=False)
        key = c.make_key("http://x", None)
        c.set(key, {"r": 1})
        assert c.get(key) is None

    def test_ttl_expiry(self, monkeypatch):
        c = APICache(ttl_seconds=10)
        key = c.make_key("http://x", None)
        c.set(key, {"r": 1})
        # advance time far past TTL (well beyond the real timestamp stored on set)
        monkeypatch.setattr(http_mod.time, "time", lambda: 10_000_000_000)
        assert c.get(key) is None

    def test_corrupt_cache_returns_none(self):
        c = APICache()
        key = c.make_key("http://x", None)
        path = c._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json")
        assert c.get(key) is None


class TestRateLimiter:
    def test_acquire_with_tokens_no_wait(self):
        rl = RateLimiter(requests_per_second=10.0, burst_size=5)
        assert rl.acquire() == 0.0

    def test_acquire_blocks_when_empty(self, monkeypatch):
        rl = RateLimiter(requests_per_second=1000.0, burst_size=1)
        sleeps = []
        monkeypatch.setattr(http_mod.time, "sleep", lambda s: sleeps.append(s))
        rl.acquire()  # consume the one token
        waited = rl.acquire(tokens=1)  # must wait
        assert waited > 0
        assert sleeps  # slept

    def test_try_acquire(self):
        rl = RateLimiter(requests_per_second=0.0001, burst_size=1)
        assert rl.try_acquire() is True
        assert rl.try_acquire() is False


class TestAPIClientGet:
    def _client(self, **kw):
        client = APIClient(base_url="https://api.example.com", **kw)
        client.session = MagicMock()
        return client

    def test_get_success(self):
        client = self._client()
        client.session.get.return_value = make_response(json_data={"ok": True})
        data = client.get("/endpoint", params={"q": "x"})
        assert data == {"ok": True}
        # url built from base + endpoint
        called_url = client.session.get.call_args.args[0]
        assert called_url == "https://api.example.com/endpoint"

    def test_get_no_base_url(self):
        client = APIClient()
        client.session = MagicMock()
        client.session.get.return_value = make_response(json_data={"a": 1})
        client.get("https://full.url/path")
        assert client.session.get.call_args.args[0] == "https://full.url/path"

    def test_get_http_error_raises(self):
        client = self._client()
        client.session.get.return_value = make_response(status_ok=False)
        import requests

        with pytest.raises(requests.HTTPError):
            client.get("/bad")

    def test_get_uses_and_populates_cache(self):
        cache = APICache()
        client = self._client(cache=cache)
        client.session.get.return_value = make_response(json_data={"v": 1})
        # first call -> network, cached
        assert client.get("/e", params={"a": 1}) == {"v": 1}
        # second call should hit cache, not call session again
        client.session.get.reset_mock()
        assert client.get("/e", params={"a": 1}) == {"v": 1}
        client.session.get.assert_not_called()

    def test_get_bypass_cache(self):
        cache = APICache()
        client = self._client(cache=cache)
        client.session.get.return_value = make_response(json_data={"v": 1})
        client.get("/e", use_cache=False)
        client.get("/e", use_cache=False)
        assert client.session.get.call_count == 2

    def test_get_with_rate_limiter(self, monkeypatch):
        rl = RateLimiter(requests_per_second=1000.0)
        client = self._client(rate_limiter=rl)
        monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)
        client.session.get.return_value = make_response(json_data={"ok": 1})
        assert client.get("/e") == {"ok": 1}


class TestAPIClientPost:
    def test_post_success(self):
        client = APIClient(base_url="https://api.example.com")
        client.session = MagicMock()
        client.session.post.return_value = make_response(json_data={"created": True})
        result = client.post("/items", json_data={"name": "x"})
        assert result == {"created": True}

    def test_post_with_rate_limiter(self):
        rl = RateLimiter(requests_per_second=1000.0)
        client = APIClient(rate_limiter=rl)
        client.session = MagicMock()
        client.session.post.return_value = make_response(json_data={"ok": 1})
        assert client.post("/x", data={"f": 1}) == {"ok": 1}

    def test_post_http_error(self):
        client = APIClient()
        client.session = MagicMock()
        client.session.post.return_value = make_response(status_ok=False)
        import requests

        with pytest.raises(requests.HTTPError):
            client.post("/x")


class TestAPIClientDownload:
    def test_download_writes_file(self, tmp_path):
        client = APIClient()
        client.session = MagicMock()
        client.session.get.return_value = make_response(
            content_chunks=[b"abc", b"def"],
            headers={"content-length": "6"},
        )
        dest = tmp_path / "sub" / "file.bin"
        out = client.download("http://x/file.bin", dest)
        assert out == dest
        assert dest.read_bytes() == b"abcdef"

    def test_download_progress_callback(self, tmp_path):
        client = APIClient()
        client.session = MagicMock()
        client.session.get.return_value = make_response(
            content_chunks=[b"ab", b"cd"],
            headers={"content-length": "4"},
        )
        seen = []
        client.download(
            "http://x/f.bin",
            tmp_path / "f.bin",
            progress_callback=lambda d, t: seen.append((d, t)),
        )
        assert seen == [(2, 4), (4, 4)]

    def test_download_http_error(self, tmp_path):
        client = APIClient()
        client.session = MagicMock()
        client.session.get.return_value = make_response(status_ok=False)
        import requests

        with pytest.raises(requests.HTTPError):
            client.download("http://x/f.bin", tmp_path / "f.bin")

    def test_download_with_rate_limiter(self, tmp_path):
        rl = RateLimiter(requests_per_second=1000.0)
        client = APIClient(rate_limiter=rl)
        client.session = MagicMock()
        client.session.get.return_value = make_response(
            content_chunks=[b"x"], headers={}
        )
        out = client.download("http://x/f", tmp_path / "f")
        assert out.read_bytes() == b"x"


class TestContextManagerAndClose:
    def test_context_manager_closes(self):
        with APIClient() as client:
            client.session = MagicMock()
            sess = client.session
        sess.close.assert_called_once()

    def test_close(self):
        client = APIClient()
        client.session = MagicMock()
        client.close()
        client.session.close.assert_called_once()


class TestConfigAwareDecorators:
    def test_config_aware_fills_default(self, monkeypatch):
        from bioamla.common.config import Config, set_config

        set_config(Config(audio={"sample_rate": 8000}))

        @config_aware("audio")
        def fn(sample_rate: int | None = None):
            return sample_rate

        try:
            assert fn() == 8000  # filled from config
            assert fn(sample_rate=22050) == 22050  # explicit wins
        finally:
            from bioamla.common.config import reset_config

            reset_config()

    def test_config_aware_custom_mapping(self):
        from bioamla.common.config import Config, reset_config, set_config

        set_config(Config(inference={"top_k": 3}))

        @config_aware("inference", mapping={"num_results": "top_k"})
        def predict(num_results: int | None = None):
            return num_results

        try:
            assert predict() == 3
        finally:
            reset_config()

    def test_config_aware_missing_key_keeps_none(self):
        from bioamla.common.config import Config, reset_config, set_config

        set_config(Config(audio={}))

        @config_aware("audio")
        def fn(sample_rate: int | None = None):
            return sample_rate

        try:
            assert fn() is None
        finally:
            reset_config()

    def test_config_aware_class(self):
        from bioamla.common.config import Config, reset_config, set_config

        set_config(Config(audio={"sample_rate": 16000}))

        @config_aware_class("audio")
        class Proc:
            def run(self, sample_rate: int | None = None):
                return sample_rate

        try:
            assert Proc().run() == 16000
        finally:
            reset_config()

    def test_config_aware_mixin(self):
        from bioamla.common.config import Config, reset_config, set_config

        set_config(Config(audio={"sample_rate": 44100, "mono": True}))

        class Proc(ConfigAwareMixin):
            _config_section = "audio"

            def run(self, sample_rate=None):
                return self._get_config_default("sample_rate", sample_rate)

        try:
            p = Proc()
            assert p.run() == 44100
            assert p.run(8000) == 8000
            defaults = p._get_config_defaults(sample_rate=None, mono=False)
            assert defaults["sample_rate"] == 44100
            assert defaults["mono"] is False
        finally:
            reset_config()


class TestRateLimitedDecorator:
    def test_rate_limited(self):
        calls = []

        @rate_limited(1000.0)
        def fn():
            calls.append(1)
            return "done"

        assert fn() == "done"
        assert calls == [1]
