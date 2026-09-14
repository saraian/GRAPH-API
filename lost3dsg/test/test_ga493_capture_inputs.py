"""The GA-493 capture records the two inputs the replay needs, and names its embedder.

Three inputs decide association and only one of them was recorded before this test:

- a pose, without which `_record_sighting` abstains and every candidate search silently
  falls back to the fixed 1.0 m radius;
- the wall segments, whose room polygons set `room_id` on every admission;
- the embedding model, which was asked for by path and never checked for identity.

The embedder branch is the one with three answers. `SemanticEmbedder.__init__` tries
SentenceTransformer, falls back to Word2Vec, and continues on either failure, so
"sentence_transformer", "word2vec" and "none" are three different systems that a blank
manifest field could not tell apart.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_manager_6 as om6  # noqa: E402


class Recorder:
    """Stands in for ReplayCapture: keeps what was recorded, in order."""

    def __init__(self):
        self.events = []

    def event(self, kind, payload=None, cycle_id=None):
        self.events.append((kind, payload or {}))


def kinds(recorder):
    return [kind for kind, _ in recorder.events]


def payload_of(recorder, kind):
    return next(payload for name, payload in recorder.events if name == kind)


def test_embedding_identity_names_the_component_that_answered():
    saved_st = om6.world2vec.st_model
    saved_w2v = om6.world2vec.w2v_model
    try:
        om6.world2vec.st_model = object()
        om6.world2vec.w2v_model = None
        assert _embedding_identity()["loaded"].startswith("sentence_transformer:")

        om6.world2vec.st_model = None
        om6.world2vec.w2v_model = object()
        assert _embedding_identity()["loaded"].startswith("word2vec:")

        # Neither loaded is NOT the same as "word2vec, configured". This is the case the
        # old blank field could not distinguish, and the one that changes every score.
        om6.world2vec.st_model = None
        om6.world2vec.w2v_model = None
        assert _embedding_identity()["loaded"] == "none"
    finally:
        om6.world2vec.st_model = saved_st
        om6.world2vec.w2v_model = saved_w2v


def _embedding_identity():
    identity = om6._embedding_model_identity()
    # The configured path is reported whatever answered, so "asked for" and "answered"
    # stay separable (working rule 4: written vs tested, asked vs answered).
    assert "word2vec_path_configured" in identity
    assert isinstance(identity["word2vec_path_exists"], bool)
    return identity


def test_agent_pose_is_recorded_with_its_geometry():
    node = object.__new__(om6.ObjectManagerService)
    node.ga493_replay_capture = Recorder()
    node.agent_poses = []
    node.agent_pose_history = []
    node.latest_agent_pose = None
    node._last_pose_snapshot = node._last_path_publish = 1e12
    node.agent_path_pub = rosstub.Any()
    node.room_manager = NS(_effective_room_id=lambda: "room_1")
    node._room_frames_for = lambda room_id: None

    msg = NS(
        header=NS(stamp=NS(sec=7, nanosec=500_000_000), frame_id="map"),
        pose=NS(position=NS(x=1.5, y=-2.0, z=0.25),
                orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)),
    )
    om6.ObjectManagerService._agent_pose_callback(node, msg)

    assert kinds(node.ga493_replay_capture) == ["agent_pose_arrived"]
    pose = payload_of(node.ga493_replay_capture, "agent_pose_arrived")["pose"]
    assert (pose["x"], pose["y"], pose["z"]) == (1.5, -2.0, 0.25)
    assert pose["timestamp"] == 7.5
    assert pose["reference_frame"] == "map"


def test_walls_are_recorded_as_segments_not_as_the_rooms_they_build():
    node = object.__new__(om6.ObjectManagerService)
    node.ga493_replay_capture = Recorder()
    ingested = []
    node.room_manager = NS(
        ingest_detected_walls=ingested.append,
        _effective_room_id=lambda: "room_2",
    )

    segments = [{"start": {"x": 0.0, "y": 0.0}, "end": {"x": 3.0, "y": 0.0}}]
    om6.ObjectManagerService.walls_callback(node, NS(data=om6.json.dumps(segments)))

    assert ingested == [segments]
    recorded = payload_of(node.ga493_replay_capture, "walls_arrived")
    # The SEGMENTS, so the replay re-derives the rooms. Handing it the polygons would
    # freeze the decision the replay exists to re-derive.
    assert recorded["segments"] == segments
    assert recorded["room_id_after"] == "room_2"


def test_capture_disabled_records_nothing_and_does_not_crash():
    node = object.__new__(om6.ObjectManagerService)
    node.ga493_replay_capture = None
    ingested = []
    node.room_manager = NS(ingest_detected_walls=ingested.append,
                           _effective_room_id=lambda: "room_3")
    om6.ObjectManagerService.walls_callback(node, NS(data="[]"))
    assert ingested == [[]]


if __name__ == "__main__":
    test_embedding_identity_names_the_component_that_answered()
    test_agent_pose_is_recorded_with_its_geometry()
    test_walls_are_recorded_as_segments_not_as_the_rooms_they_build()
    test_capture_disabled_records_nothing_and_does_not_crash()
    print("OK: pose and walls are captured, and the manifest names the embedder that answered")
