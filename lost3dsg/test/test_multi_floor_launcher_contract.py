import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
LAUNCH = (ROOT / "lost3dsg/launch/habitat_launch.py").read_text()
STACK = (ROOT / "lost3dsg/test/live_stack_container.sh").read_text()
RUNNER = (ROOT / "run_sim.sh").read_text()
FEED = (ROOT / "lost3dsg/test/habitat_feed_host.py").read_text()
FEED_NODE = (ROOT / "lost3dsg/src/perception_module/habitat_feed_node.py").read_text()


class MultiFloorLauncherContractTest(unittest.TestCase):
    def test_launch_exposes_database_and_session_mode(self):
        self.assertIn("'rtabmap_session_mode'", LAUNCH)
        self.assertIn("'rtabmap_database_path'", LAUNCH)
        self.assertIn("'localization': use_existing_rtabmap_database", LAUNCH)
        self.assertIn("'database_path': rtabmap_database_path", LAUNCH)
        self.assertNotIn("'database_path': '/root/.ros/rtabmap.db'", LAUNCH)

    def test_delete_is_mapping_only(self):
        expression = LAUNCH[LAUNCH.index("rtabmap_args = PythonExpression"):]
        expression = expression[:expression.index("# ------------------------------------------------------------")]
        self.assertIn("--delete_db_on_start", expression)
        self.assertIn("== 'localization'", expression)

    def test_stack_refuses_unidentified_or_mismatched_localization_map(self):
        self.assertIn('RTABMAP_SESSION_MODE="${RTABMAP_SESSION_MODE:-mapping}"', STACK)
        self.assertIn("localization database refused", STACK)
        self.assertIn("MAP PARAMETER MISMATCH", STACK)
        self.assertIn('rtabmap_session_mode:="$RTABMAP_SESSION_MODE"', STACK)
        self.assertIn('rtabmap_database_path:="$RTABMAP_DATABASE_PATH"', STACK)

    def test_runner_copies_explicit_map_and_preserves_default_mapping(self):
        self.assertIn('LOCALIZE_DB_SOURCE="${RTABMAP_LOCALIZE_DB:-}"', RUNNER)
        self.assertIn('cp --reflink=auto "$LOCALIZE_DB_SOURCE" "$LOCALIZE_DB_COPY"', RUNNER)
        self.assertIn("export RTABMAP_SESSION_MODE=localization", RUNNER)
        self.assertIn("export RTABMAP_SESSION_MODE=mapping", RUNNER)
        self.assertIn('unset RTABMAP_LOCALIZE_DB', RUNNER)
        self.assertNotIn('rm -f "$LOCALIZE_DB_COPY"', RUNNER)

    def test_container_receives_complete_database_contract(self):
        docker = RUNNER[RUNNER.index("docker run --name graphapi_live"):]
        self.assertIn("-e RTABMAP_LOCALIZE_DB", docker)
        self.assertIn("-e RTABMAP_SESSION_MODE", docker)
        self.assertIn("-e RTABMAP_DATABASE_PATH", docker)

    def test_feed_exposes_persistent_session_identity(self):
        self.assertIn("def main(sim=None, session_context=None, runtime=None):", FEED)
        self.assertIn("**FLOOR_SESSION_CONTEXT", FEED)
        self.assertIn('runtime["object_controller"] = object_controller', FEED)
        self.assertIn('runtime["belief_poller"] = poller', FEED)
        driver = (ROOT / "lost3dsg/test/persistent_habitat_feed.py").read_text()
        self.assertIn('runtime["observation_guard"]', driver)
        self.assertIn("import habitat_feed_host as feed", driver)

    def test_runner_exposes_explicit_persistent_multi_floor_mode(self):
        self.assertIn("--multi-floor", RUNNER)
        self.assertIn("MULTI_FLOOR_SEQUENCE", RUNNER)
        self.assertIn("MULTI_FLOOR_TRANSFORMS", RUNNER)
        self.assertIn("persistent_habitat_feed.py", RUNNER)
        self.assertIn('"$MULTI_FLOOR_COORD_DIR/closed.json"', RUNNER)
        self.assertIn('"multi_floor_visits": visits', RUNNER)
        self.assertIn('"multi_floor_schedule":', RUNNER)
        self.assertIn('"multi_floor_transforms":', RUNNER)
        self.assertIn('"dynamic_actions_sha256":', RUNNER)
        self.assertIn("refusing to overwrite visit activation record", RUNNER)
        self.assertIn("refusing to overwrite visit close record", RUNNER)
        self.assertIn('_multi_config_localization=$(python3 -c', RUNNER)
        self.assertIn('export FEED_POSE_SOURCE="$_multi_pose_source"', RUNNER)

    def test_ros_ingress_rejects_cross_floor_frames_before_publication(self):
        self.assertIn('self._expected_floor_session', FEED_NODE)
        guard = FEED_NODE[FEED_NODE.index('expected = self._expected_floor_session'):]
        self.assertIn('refusing stale or cross-floor feed frame', guard)
        self.assertLess(
            FEED_NODE.index('expected = self._expected_floor_session'),
            FEED_NODE.index('stamp = self.get_clock().now().to_msg()',
                            FEED_NODE.index('expected = self._expected_floor_session')),
        )
        docker = RUNNER[RUNNER.index("docker run --name graphapi_live"):]
        self.assertIn("-e MULTI_FLOOR_SESSION_ID", docker)
        self.assertIn("-e MULTI_FLOOR_FLOOR_ID", docker)
        self.assertIn("-e MULTI_FLOOR_VISIT_INDEX", docker)


if __name__ == "__main__":
    unittest.main()
