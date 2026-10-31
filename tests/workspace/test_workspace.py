"""The .bioamla/ workspace format."""

import json
import shutil
from pathlib import Path

import pytest

from bioamla.exceptions import InvalidInputError, NotFoundError, WorkspaceError
from bioamla.workspace import (
    FORMAT_VERSION,
    WORKSPACE_DIR,
    create_workspace,
    find_workspace,
    is_workspace,
    open_workspace,
)


@pytest.fixture
def ws(tmp_path):
    return create_workspace(tmp_path / "survey")


class TestCreateAndOpen:
    def test_layout(self, ws) -> None:
        dot = ws.root / WORKSPACE_DIR
        for sub in ("annotations", "imported", "datasets", "models", "runs"):
            assert (dot / sub).is_dir()
        assert (dot / "workspace.toml").is_file()
        assert ws.artifacts() == []
        assert ws.name == "survey" and ws.format_version == FORMAT_VERSION

    def test_round_trip(self, ws) -> None:
        again = open_workspace(ws.root)
        assert (again.id, again.name, again.created) == (ws.id, ws.name, ws.created)

    def test_name_with_quotes(self, tmp_path) -> None:
        ws = create_workspace(tmp_path / "x", name='Costa Rica "2024" survey')
        assert open_workspace(ws.root).name == 'Costa Rica "2024" survey'

    def test_create_twice_fails(self, ws) -> None:
        with pytest.raises(WorkspaceError, match="already a workspace"):
            create_workspace(ws.root)

    def test_open_non_workspace(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Not a bioamla workspace"):
            open_workspace(tmp_path)

    def test_newer_format_refused(self, ws) -> None:
        meta = ws.dir / "workspace.toml"
        meta.write_text(meta.read_text().replace(f"format_version = {FORMAT_VERSION}",
                                                 f"format_version = {FORMAT_VERSION + 1}"))  # fmt: skip
        with pytest.raises(WorkspaceError, match="newer than this bioamla"):
            open_workspace(ws.root)

    def test_garbled_metadata(self, ws) -> None:
        (ws.dir / "workspace.toml").write_text("not = [toml")
        with pytest.raises(WorkspaceError, match="Unreadable"):
            open_workspace(ws.root)


class TestFind:
    def test_walks_up(self, ws) -> None:
        deep = ws.root / "a" / "b"
        deep.mkdir(parents=True)
        assert find_workspace(deep).id == ws.id

    def test_none_outside(self, tmp_path) -> None:
        assert find_workspace(tmp_path) is None

    def test_cwd_default(self, ws, monkeypatch) -> None:
        monkeypatch.chdir(ws.root)
        assert find_workspace().id == ws.id

    def test_is_workspace(self, ws, tmp_path) -> None:
        assert is_workspace(ws.root) and not is_workspace(tmp_path)


class TestArtifacts:
    def test_link_inside_is_relative(self, ws) -> None:
        rec = ws.root / "recordings" / "a.wav"
        rec.parent.mkdir()
        rec.write_bytes(b"RIFF")
        artifact = ws.link(rec)
        assert artifact.path == "recordings/a.wav"
        assert artifact.kind == "audio_file" and artifact.mode == "linked"
        assert ws.resolve(artifact) == rec

    def test_link_outside_is_absolute(self, ws, tmp_path) -> None:
        archive = tmp_path / "archive"
        archive.mkdir()
        artifact = ws.link(archive)
        assert artifact.kind == "folder"
        assert Path(artifact.path).is_absolute()
        assert ws.resolve(artifact) == archive

    def test_relative_links_survive_moving_the_workspace(self, ws, tmp_path) -> None:
        (ws.root / "a.wav").write_bytes(b"RIFF")
        ws.link(ws.root / "a.wav")
        moved = tmp_path / "moved"
        shutil.move(str(ws.root), moved)
        reopened = open_workspace(moved)
        (artifact,) = reopened.artifacts()
        assert reopened.resolve(artifact) == moved / "a.wav"
        assert reopened.resolve(artifact).exists()

    def test_link_keeps_symlinks(self, ws, tmp_path) -> None:
        real = tmp_path / "vault" / "audio"
        real.mkdir(parents=True)
        link = tmp_path / "audio-link"
        link.symlink_to(real)
        assert ws.link(link).path == str(link)

    def test_link_twice_is_one_artifact(self, ws, tmp_path) -> None:
        folder = tmp_path / "f"
        folder.mkdir()
        assert ws.link(folder).id == ws.link(folder).id
        assert len(ws.artifacts()) == 1

    def test_link_missing(self, ws, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            ws.link(tmp_path / "nope.wav")

    def test_bad_kind(self, ws, tmp_path) -> None:
        (tmp_path / "x").write_text("x")
        with pytest.raises(InvalidInputError, match="Unknown artifact kind"):
            ws.link(tmp_path / "x", kind="spreadsheet")

    def test_import_copies(self, ws, tmp_path) -> None:
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF1")
        artifact = ws.import_copy(src)
        assert artifact.mode == "imported" and artifact.path == "a.wav"
        assert ws.resolve(artifact).read_bytes() == b"RIFF1"
        src.write_bytes(b"RIFF2")  # the copy is independent
        assert ws.resolve(artifact).read_bytes() == b"RIFF1"

    def test_import_name_clash_does_not_overwrite(self, ws, tmp_path) -> None:
        (tmp_path / "one").mkdir()
        (tmp_path / "two").mkdir()
        (tmp_path / "one" / "a.wav").write_bytes(b"1")
        (tmp_path / "two" / "a.wav").write_bytes(b"2")
        first = ws.import_copy(tmp_path / "one" / "a.wav")
        second = ws.import_copy(tmp_path / "two" / "a.wav")
        assert first.path != second.path
        assert ws.resolve(first).read_bytes() == b"1" and ws.resolve(second).read_bytes() == b"2"

    def test_import_folder(self, ws, tmp_path) -> None:
        folder = tmp_path / "clips"
        folder.mkdir()
        (folder / "c.wav").write_bytes(b"RIFF")
        artifact = ws.import_copy(folder, kind="dataset")
        assert artifact.kind == "dataset"
        assert (ws.resolve(artifact) / "c.wav").exists()

    def test_filter_get_remove(self, ws, tmp_path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        (tmp_path / "m").mkdir()
        audio = ws.link(tmp_path / "a.wav")
        model = ws.link(tmp_path / "m", kind="model")
        assert ws.artifacts(kind="model") == [model]
        assert ws.get_artifact(audio.id) == audio
        ws.remove(audio.id)
        assert ws.artifacts() == [model]
        assert (tmp_path / "a.wav").exists()  # linked files are never touched
        with pytest.raises(NotFoundError):
            ws.get_artifact(audio.id)

    def test_remove_import_can_delete_copy(self, ws, tmp_path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        artifact = ws.import_copy(tmp_path / "a.wav")
        copy = ws.resolve(artifact)
        ws.remove(artifact.id, delete_import=True)
        assert not copy.exists()

    def test_garbled_artifacts(self, ws) -> None:
        (ws.dir / "artifacts.json").write_text("{nope")
        with pytest.raises(WorkspaceError, match="Unreadable"):
            ws.artifacts()

    def test_to_dict(self, ws, tmp_path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        ws.link(tmp_path / "a.wav")
        d = ws.to_dict()
        json.dumps(d)
        assert d["artifacts"] == {"audio_file": 1} and d["name"] == "survey"


def test_import_bioamla_stays_light() -> None:
    import subprocess
    import sys

    code = "import sys, bioamla.workspace; print('numpy' in sys.modules, 'pydantic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


class TestSafety:
    @pytest.mark.parametrize("bad", ["", "/etc", "../../outside", "sub/../../x"])
    def test_imported_paths_must_stay_inside(self, ws, tmp_path, bad) -> None:
        victim = tmp_path / "keep"
        victim.mkdir()
        (ws.dir / "artifacts.json").write_text(json.dumps(
            {"artifacts": [{"id": "x", "kind": "folder", "path": bad, "mode": "imported"}]}
        ))  # fmt: skip
        with pytest.raises(WorkspaceError, match="outside imported/"):
            ws.remove("x", delete_import=True)
        assert victim.exists() and ws.imported_dir.exists()
        assert [a.id for a in ws.artifacts()] == ["x"]  # nothing was forgotten either

    @pytest.mark.parametrize("content", ["[]", '"x"', '{"artifacts": {}}', '{"artifacts": [1]}'])
    def test_wrong_shape_is_a_workspace_error(self, ws, content) -> None:
        (ws.dir / "artifacts.json").write_text(content)
        with pytest.raises(WorkspaceError, match="Unreadable"):
            ws.artifacts()

    def test_unknown_mode_rejected(self, ws) -> None:
        (ws.dir / "artifacts.json").write_text(json.dumps(
            {"artifacts": [{"id": "x", "kind": "file", "path": "a", "mode": "copy"}]}
        ))  # fmt: skip
        with pytest.raises(WorkspaceError, match="unknown mode 'copy'"):
            ws.artifacts()

    def test_cannot_import_the_workspace_into_itself(self, ws) -> None:
        with pytest.raises(InvalidInputError, match="contains this workspace"):
            ws.import_copy(ws.root)
        with pytest.raises(InvalidInputError, match="contains this workspace"):
            ws.import_copy(ws.root.parent)

    def test_failed_import_leaves_no_copy(self, ws, tmp_path) -> None:
        (ws.dir / "artifacts.json").write_text("{nope")
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        with pytest.raises(WorkspaceError):
            ws.import_copy(tmp_path / "a.wav")
        assert not any(ws.imported_dir.iterdir())

    def test_config_file_created(self, ws) -> None:
        assert ws.config_path.is_file()


def _link_many(args) -> None:
    root, files = args
    from bioamla.workspace import open_workspace as _open

    ws = _open(root)
    for f in files:
        ws.link(f)


def test_concurrent_links_are_all_kept(tmp_path) -> None:
    import multiprocessing

    ws = create_workspace(tmp_path / "ws")
    jobs = []
    for p in range(8):
        files = []
        for i in range(10):
            f = tmp_path / f"p{p}_{i}.wav"
            f.write_bytes(b"RIFF")
            files.append(str(f))
        jobs.append((str(ws.root), files))
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        pool.map(_link_many, jobs)
    assert len(ws.artifacts()) == 80
