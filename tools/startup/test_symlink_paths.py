"""Host symlink targets must survive every Docker launch boundary."""
from pathlib import Path

import pytest

from graphapi_cli.bootstrap import mount_plan
from graphapi_cli.configuration import ConfigurationError
from graphapi_cli.docker import mounted_paths
from graphapi_cli.paths import host_path, output_directory


@pytest.mark.parametrize("name", ["results", "maps", "schedules", "ws"])
@pytest.mark.parametrize("kind", ["absolute", "relative", "chain"])
def test_planner_follows_output_links_in_host_namespace(tmp_path, name, kind):
    host = tmp_path / "host"
    checkout = host / "repo"
    (checkout / "config").mkdir(parents=True)
    target = host / "storage" / name
    target.mkdir(parents=True)
    (host / "home/user/cache").mkdir(parents=True)
    if kind == "absolute":
        link = Path("/storage") / name
    elif kind == "relative":
        link = Path("../storage") / name
    else:
        (host / "storage/alias").symlink_to(name)
        link = Path("/storage/alias")
    (checkout / name).symlink_to(link, target_is_directory=True)
    _, _, args = mount_plan(Path("/repo"), ["status"],
                           {"HOME": "/home/user", "XDG_CACHE_HOME": "/home/user/cache"},
                           host_view=True, host_root=host)
    assert f"/storage/{name}:/storage/{name}" in args
    assert (checkout / name).is_symlink()
    assert list(target.iterdir()) == []  # Planning must not create run artifacts.


def test_host_namespace_symlink_loop_is_reported(tmp_path):
    (tmp_path / "a").symlink_to("/b")
    (tmp_path / "b").symlink_to("/a")
    with pytest.raises(ConfigurationError, match="too many symlinks"):
        host_path("/a", tmp_path)


def test_missing_output_link_target_can_be_created_without_replacing_link(tmp_path):
    target = tmp_path / "storage/new/results"
    link = tmp_path / "results"
    link.symlink_to(target, target_is_directory=True)
    assert output_directory(link) == target
    assert link.is_symlink() and link.is_dir()


def test_application_mounts_preserve_external_outputs_and_inputs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    storage = tmp_path / "storage"
    output = storage / "results"
    models = storage / "models"
    output.mkdir(parents=True)
    models.mkdir()
    (root / "results").symlink_to(output, target_is_directory=True)
    (root / "models").symlink_to(models, target_is_directory=True)
    mounts = mounted_paths(root, {"WORKSPACE_ROOT": str(root), "SAM_MODEL_DIR": str(root / "models")})
    assert mounts[str(output)] is True
    assert mounts[str(models)] is False
    assert (root / "results").is_symlink()


def test_output_path_that_is_a_file_still_fails(tmp_path):
    path = tmp_path / "results"
    path.write_text("not a directory")
    with pytest.raises(FileExistsError):
        output_directory(path)
