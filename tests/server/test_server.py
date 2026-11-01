"""bioamla serve: the HTTP API, its token and its path policy."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bioamla.exceptions import InvalidInputError, NotFoundError  # noqa: E402
from bioamla.server import (  # noqa: E402
    AccessDenied,
    Identity,
    PathPolicy,
    ServerConfig,
    create_app,
    read_or_create_token,
)

TOKEN = "s3cret-token"
ME = Identity()


def _can_symlink(tmp_path) -> bool:
    try:
        (tmp_path / ".probe-link").symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    (tmp_path / ".probe-link").unlink()
    return True


@pytest.fixture
def symlinks(tmp_path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("can't create symlinks here")


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    return TestClient(create_app(ServerConfig(root=root, token=TOKEN)))


def auth(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestRoutes:
    def test_health_needs_no_token(self, client) -> None:
        body = client.get("/v1/health").json()
        assert body["ok"] is True and body["result"] == {"status": "ok"}

    def test_version_needs_the_token(self, client) -> None:
        missing = client.get("/v1/version")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        assert missing.json()["error"]["type"] == "Unauthenticated"
        assert client.get("/v1/version", headers=auth("wrong")).status_code == 401
        assert client.get("/v1/version", headers={"Authorization": TOKEN}).status_code == 401

    def test_version(self, client) -> None:
        result = client.get("/v1/version", headers=auth()).json()["result"]
        assert result["api"] == "v1" and result["workspace_format"] == 1
        assert result["user"] == "owner"

    def test_openapi_needs_the_token(self, client) -> None:
        missing = client.get("/v1/openapi.json")
        assert missing.status_code == 401 and missing.json()["ok"] is False
        spec = client.get("/v1/openapi.json", headers=auth()).json()
        assert "/v1/version" in spec["paths"]
        assert "/v1/openapi.json" not in spec["paths"]
        # generated clients learn that routes need the bearer token
        assert spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
        assert spec["paths"]["/v1/version"]["get"]["security"] == [{"HTTPBearer": []}]
        assert "security" not in spec["paths"]["/v1/health"]["get"]

    @pytest.mark.parametrize("path", ["/v1/docs", "/v1/redoc", "/docs"])
    def test_no_docs_pages(self, client, path) -> None:
        assert client.get(path).status_code == 404

    def test_routing_errors_use_the_envelope(self, client) -> None:
        missing = client.get("/v1/nope", headers=auth())
        assert missing.status_code == 404
        assert missing.json()["ok"] is False
        assert missing.json()["error"]["type"] == "NotFoundError"
        wrong = client.post("/v1/version", headers=auth())
        assert wrong.status_code == 405 and wrong.headers["allow"] == "GET"
        assert wrong.json()["error"]["type"] == "MethodNotAllowed"

    def test_unexpected_errors_use_the_envelope(self, tmp_path) -> None:
        app = create_app(ServerConfig(root=tmp_path, token=TOKEN))

        @app.get("/v1/boom")
        def boom() -> None:
            raise RuntimeError("secret detail")

        @app.get("/v1/needs-int")
        def needs_int(n: int) -> dict:
            return {"n": n}

        c = TestClient(app, raise_server_exceptions=False)
        body = c.get("/v1/boom").json()
        assert body["ok"] is False and body["error"]["type"] == "InternalError"
        assert "secret detail" not in str(body)
        bad = c.get("/v1/needs-int?n=x")
        assert bad.status_code == 422 and bad.json()["error"]["type"] == "ValidationError"


class TestPathPolicy:
    @pytest.fixture
    def policy(self, tmp_path):
        (tmp_path / "root" / "ws").mkdir(parents=True)
        (tmp_path / "archive").mkdir()
        (tmp_path / "private").mkdir()
        return PathPolicy(tmp_path / "root", [tmp_path / "archive"])

    def test_inside_root_and_linkable(self, policy, tmp_path) -> None:
        assert policy.authorize(ME, "ws") == (tmp_path / "root" / "ws").resolve()
        assert policy.authorize(ME, tmp_path / "archive" / "a.wav").name == "a.wav"

    @pytest.mark.parametrize("path", ["../private", "ws/../../private", "/etc/passwd"])
    def test_outside_is_denied(self, policy, path) -> None:
        with pytest.raises(AccessDenied):
            policy.authorize(ME, path)

    def test_symlinks_cant_escape(self, policy, tmp_path, symlinks) -> None:
        (tmp_path / "root" / "ws" / "sneaky").symlink_to(tmp_path / "private")
        with pytest.raises(AccessDenied):
            policy.authorize(ME, "ws/sneaky/x")

    def test_denied_paths_are_403_and_bad_tokens_401(self, client) -> None:
        from fastapi import Depends

        app = client.app

        @app.get("/v1/check")
        def check(path: str, who: Identity = Depends(app.state.identity)) -> dict:
            app.state.policy.authorize(who, path)
            return {}

        outside = {"path": "../token_backup"}
        response = client.get("/v1/check", params=outside, headers=auth())
        assert response.status_code == 403 and response.json()["error"]["type"] == "AccessDenied"
        assert client.get("/v1/check", params=outside).status_code == 401
        assert client.get("/v1/check", params=outside, headers=auth("wrong")).status_code == 401
        assert client.get("/v1/check", params={"path": "ok"}, headers=auth()).status_code == 200

    def test_client_paths_are_literal(self, policy) -> None:
        # "~" is a folder name under the root here, never the server user's home
        assert policy.authorize(ME, "~/x") == policy.root / "~" / "x"
        with pytest.raises(InvalidInputError):
            policy.authorize(ME, "a\x00b")

    def test_workspace_cant_be_a_link_out_of_the_root(self, policy, tmp_path, symlinks) -> None:
        from bioamla.workspace import create_workspace

        create_workspace(tmp_path / "archive" / "survey")
        (tmp_path / "root" / "alias").symlink_to(tmp_path / "archive" / "survey")
        with pytest.raises(AccessDenied, match="not inside the workspace root"):
            policy.workspace_dir(ME, "alias")

    @pytest.mark.parametrize("linked", [".bioamla", ".bioamla/workspace.toml"])
    def test_workspace_metadata_cant_be_a_link_out(
        self, policy, tmp_path, symlinks, linked
    ) -> None:
        from bioamla.workspace import create_workspace

        create_workspace(tmp_path / "private" / "real")
        (tmp_path / "root" / "survey" / ".bioamla").mkdir(parents=True)
        target = tmp_path / "private" / "real" / linked
        link = tmp_path / "root" / "survey" / linked
        if link.exists():
            link.rmdir()
        link.symlink_to(target)
        with pytest.raises(AccessDenied, match="metadata outside"):
            policy.workspace_dir(ME, "survey")

    def test_workspace_names(self, policy, tmp_path) -> None:
        from bioamla.workspace import create_workspace

        create_workspace(tmp_path / "root" / "survey")
        assert policy.workspace_dir(ME, "survey").name == "survey"
        for bad in ("", ".", "..", "a/b", ".hidden", "~", "a\\b"):
            with pytest.raises(InvalidInputError):
                policy.workspace_dir(ME, bad)
        with pytest.raises(NotFoundError):
            policy.workspace_dir(ME, "ws")  # a folder, but not a workspace


class TestToken:
    def test_created_private(self, tmp_path) -> None:
        path = tmp_path / "sub" / "token"
        token = read_or_create_token(path)
        assert len(token) >= 32 and read_or_create_token(path) == token
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_readable_by_others_refused(self, tmp_path) -> None:
        path = tmp_path / "token"
        path.write_text("abc\n")
        path.chmod(0o644)
        with pytest.raises(InvalidInputError, match="accessible by other users"):
            read_or_create_token(path)

    def test_empty_refused(self, tmp_path) -> None:
        path = tmp_path / "token"
        path.write_text("\n")
        path.chmod(0o600)
        with pytest.raises(InvalidInputError, match="empty"):
            read_or_create_token(path)

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (b"one\ntwo\n", "alone, on one line"),
            (b"to ken\n", "alone, on one line"),
            (b"\xff\xfe\n", "not UTF-8"),
        ],
    )
    def test_malformed_refused(self, tmp_path, content, match) -> None:
        path = tmp_path / "token"
        path.write_bytes(content)
        path.chmod(0o600)
        with pytest.raises(InvalidInputError, match=match):
            read_or_create_token(path)

    def test_surrounding_blank_lines_are_fine(self, tmp_path) -> None:
        path = tmp_path / "token"
        path.write_text("\n  abc  \n\n")
        path.chmod(0o600)
        assert read_or_create_token(path) == "abc"

    def test_unusable_location_is_a_bioamla_error(self, tmp_path) -> None:
        from bioamla.exceptions import ConfigError

        (tmp_path / "a-file").write_text("")
        with pytest.raises(ConfigError, match="token file"):
            read_or_create_token(tmp_path / "a-file" / "token")

    def test_losing_a_creation_race_reads_the_winners_token(self, tmp_path, mocker) -> None:
        import bioamla.server._security as security

        path = tmp_path / "token"
        real_link = os.link

        def other_process_wins(src, dst):
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, b"winner\n")
            os.close(fd)
            real_link(src, dst)  # raises FileExistsError, as a lost race does

        mocker.patch.object(security.os, "link", side_effect=other_process_wins)
        assert read_or_create_token(path) == "winner"
        assert [p.name for p in tmp_path.iterdir()] == ["token"]  # no temp file left


class TestServeCommand:
    def test_starts_uvicorn_with_the_app(self, tmp_path, mocker) -> None:
        from click.testing import CliRunner

        from bioamla.cli.cli import cli

        run = mocker.patch("uvicorn.run")
        archive = tmp_path / "archive"
        archive.mkdir()
        result = CliRunner().invoke(cli, [
            "serve", "--root", str(tmp_path / "ws"), "--token-file", str(tmp_path / "t"),
            "--port", "9000", "--allow-link", str(archive),
        ])  # fmt: skip
        assert result.exit_code == 0, result.output
        app = run.call_args.args[0]
        assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 9000, "log_level": "info"}
        assert app.state.policy.linkable == (archive.resolve(),)
        assert (tmp_path / "ws").is_dir()

    @pytest.mark.skipif(
        os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions that apply"
    )
    def test_unwritable_root_is_a_clean_error(self, tmp_path, mocker) -> None:
        from click.testing import CliRunner

        from bioamla.cli.cli import cli

        mocker.patch("uvicorn.run")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            result = CliRunner().invoke(cli, ["serve", "--root", str(locked / "ws"),
                                              "--token-file", str(tmp_path / "t")])  # fmt: skip
        finally:
            locked.chmod(0o700)
        assert result.exit_code == 1 and "Can't start the server" in result.output

    def test_missing_extra_shows_the_install_hint(self, tmp_path, mocker) -> None:
        from click.testing import CliRunner

        from bioamla.cli.cli import cli
        from bioamla.exceptions import DependencyError

        mocker.patch(
            "bioamla._extras.import_extra",
            side_effect=DependencyError("uvicorn is missing: pip install 'bioamla[server]'"),
        )
        result = CliRunner().invoke(cli, ["serve", "--root", str(tmp_path / "ws"),
                                          "--token-file", str(tmp_path / "t")])  # fmt: skip
        assert result.exit_code == 1 and "bioamla[server]" in result.output

    def test_ipv6_url_has_brackets(self, tmp_path, mocker) -> None:
        from click.testing import CliRunner

        from bioamla.cli.cli import cli

        mocker.patch("uvicorn.run")
        result = CliRunner().invoke(cli, ["serve", "--root", str(tmp_path / "ws"),
                                          "--token-file", str(tmp_path / "t"), "--host", "::1"])  # fmt: skip
        assert "http://[::1]:8765/v1/" in result.output

    def test_warns_when_not_local(self, tmp_path, mocker) -> None:
        from click.testing import CliRunner

        from bioamla.cli.cli import cli

        mocker.patch("uvicorn.run")
        result = CliRunner().invoke(cli, ["serve", "--root", str(tmp_path / "ws"),
                                          "--token-file", str(tmp_path / "t"), "--host",
                                          "0.0.0.0"])  # fmt: skip
        assert result.exit_code == 0 and "VPN" in result.output


def test_server_doesnt_need_the_cli() -> None:
    import subprocess
    import sys

    code = (
        "import sys; from pathlib import Path; from fastapi.testclient import TestClient; "
        "from bioamla.server import ServerConfig, create_app; "
        "c = TestClient(create_app(ServerConfig(root=Path('.'), token='t'))); "
        "c.get('/v1/version', headers={'Authorization': 'Bearer t'}); "
        "print(sorted(m for m in ('click', 'bioamla.cli') if m in sys.modules))"
    )
    # (rich may load: pyinaturalist, a base dependency of the catalogs, imports it.)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
