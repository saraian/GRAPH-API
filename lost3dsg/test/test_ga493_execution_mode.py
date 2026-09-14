from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "lost3dsg/test/debug_configs/ga493_bbox_replay.yaml"
RUN_SIM = ROOT / "run_sim.sh"
CMAKE = ROOT / "lost3dsg/CMakeLists.txt"
DEBUG_RUN = ROOT / "lost3dsg/test/ga493_debug_run.sh"
LIVE_STACK = ROOT / "lost3dsg/test/live_stack_container.sh"
PREFLIGHT = ROOT / "lost3dsg/test/preflight_gate.py"
PERCEPTION = ROOT / "lost3dsg/src/perception_module/perception_2.py"
OBJECT_MANAGER = ROOT / "lost3dsg/src/perception_module/object_manager_6.py"


def test_debug_config_keeps_open_vocabulary_parallel_cuda_contract():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["run"]["gt_semantic"] is False
    assert config["archive"]["per_detection"] is False
    assert config["run"]["cap_min"] == 5
    assert config["habitat"]["exploration_laps"] == 1
    assert config["vlm"]["base_url"] == "https://api.regolo.ai/v1"
    assert config["vlm"]["model"] == "gemma4-31b"
    assert config["perception"]["backend"] == "modal"
    assert config["perception_parallel"]["bbox_backend"] == "cuda"
    assert config["perception_parallel"]["bbox_cuda_devices"] == [0]
    assert config["hooks"]["search_paths"] == []
    assert config["hooks"]["filter"] == ""


def test_launcher_passes_debug_identity_and_gt_refusal_inputs_to_container():
    source = RUN_SIM.read_text(encoding="utf-8")
    assert "hm3d_00824)" in source
    assert 'export GRAPH_API_RUN_ID="$RUN_ID"' in source
    assert "-e GRAPH_API_RUN_ID -e FEED_GT_SEMANTIC" in source
    assert "-e GA493_REPLAY_CAPTURE_DIR" in source


def test_capture_module_is_installed_with_ros_nodes():
    assert "src/perception_module/ga493_replay_capture.py" in CMAKE.read_text(encoding="utf-8")


def test_debug_entry_point_enforces_cap_parallel_cuda_and_no_gt():
    source = DEBUG_RUN.read_text(encoding="utf-8")
    assert 'FEED_GT_SEMANTIC=0' in source
    assert 'GRAPH_API_PARALLEL_FUSION=1' in source
    assert '"bbox_backend") == "cuda"' in source
    assert '"cap_min", ""' in source
    assert 'GA493_REPLAY_CAPTURE_DIR' in source
    assert 'GA493_CAPTURE_CLOSE_TIMEOUT' in source


def test_container_finalizes_both_capture_owners_before_map_close():
    source = LIVE_STACK.read_text(encoding="utf-8")
    cleanup = source[source.index("container_exit_cleanup()"):
                     source.index("trap container_exit_cleanup EXIT")]
    assert "_finalize_ga493_capture || true" in cleanup
    assert cleanup.index("_finalize_ga493_capture || true") < cleanup.index("_close_map_and_check")
    finalizer = source[source.index("_finalize_ga493_capture()"):
                       source.index("container_exit_cleanup()")]
    assert "producer/complete.json" in finalizer
    assert "consumer/complete.json" in finalizer
    assert "finalization_failure.txt" in finalizer
    assert 'capture owners did not start; no replay finalization is required' in finalizer
    assert "pkill -INT -f 'perception_2.py'" in finalizer
    assert "pkill -INT -f 'object_manager_6.py'" in finalizer


def test_capture_nodes_finish_cleanly_after_ros_signal_shutdown():
    perception = PERCEPTION.read_text(encoding="utf-8")
    manager = OBJECT_MANAGER.read_text(encoding="utf-8")
    assert "except Exception:" in perception
    assert "if rclpy.ok():\n            raise" in perception
    assert "from pathlib import Path as FilePath" in manager
    assert "ledger = FilePath(self.object_services._mutation_ledger_path)" in manager


def test_preflight_a12_names_each_recorder_contract_file():
    source = PREFLIGHT.read_text(encoding="utf-8")
    for path in (
        "src/perception_module/ga493_replay_capture.py",
        "test/debug_configs/ga493_bbox_replay.yaml",
        "test/ga493_debug_run.sh",
        "test/test_ga493_execution_mode.py",
        "test/test_ga493_replay_capture.py",
    ):
        assert f'"{path}"' in source
