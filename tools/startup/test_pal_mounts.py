"""A reused PAL container must execute the selected repository."""
from pathlib import Path
import pytest
from graphapi_cli.pal_mounts import checkout_source
from graphapi_cli.launch import prepare_job
from graphapi_cli.registry import Registry


def test_bag_directory_is_available_before_detaching(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    bag = tmp_path / "external" / "example"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    request = {"mode": "bag", "bag": str(bag)}
    env, state = prepare_job(root, request, inherited={"WORKSPACE_ROOT": str(tmp_path / "work"), "XDG_CACHE_HOME": str(tmp_path / "cache"), "REGOLO_API_KEY": "test-key"})
    assert env["TIAGO_BAG_DIR"] == str(bag.parent)
    Registry(tmp_path / "work").release(state["operation_id"])


def bind(source, destination):
    return {"Type": "bind", "Source": str(source), "Destination": destination}


def test_normal_checkout_mount(tmp_path):
    assert checkout_source(tmp_path, [bind(tmp_path, "/graph_api")]) == "/graph_api/lost3dsg"


def test_old_checkout_can_use_visible_current_checkout(tmp_path):
    root = tmp_path / "current"
    assert checkout_source(root, [bind(tmp_path / "old", "/graph_api"), bind(tmp_path, "/home/user")]) == "/home/user/current/lost3dsg"


def test_unavailable_checkout_fails_instead_of_using_old_sources(tmp_path):
    with pytest.raises(ValueError, match="does not mount this checkout"):
        checkout_source(tmp_path / "current", [bind(tmp_path / "old", "/graph_api")])


@pytest.mark.parametrize("hidden", ["/parent/current", "/parent/current/lost3dsg", "/parent/current/graphapi_cli"])
def test_nested_mount_hiding_checkout_is_rejected(tmp_path, hidden):
    with pytest.raises(ValueError, match="does not mount this checkout"):
        checkout_source(tmp_path / "current", [bind(tmp_path, "/parent"), bind(tmp_path / "old", hidden)])


def test_symlink_source_is_resolved(tmp_path):
    root = tmp_path / "current"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root)
    assert checkout_source(root, [bind(link, "/graph_api")]) == "/graph_api/lost3dsg"
