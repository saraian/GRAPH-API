"""A merge must not destroy the evidence it was made from.

Before 2026-09-14 the merge apply block removed the discarded object from the map and its
`observations` went with it. That list is the SOLE input of three things:
`association.position_covariance` (the object's position uncertainty),
`association.channel_covisibility` (the frames that veto a later wrong merge) and the
appearance descriptors. So consolidating two views of one object left the survivor able to
reason about it LESS well than before. The move path had always copied the list; the merge
path never did.

The survivor's centre had the same shape of defect: `Object.centroid` is assigned once, when
the object is created, and by no later code, so the box grew to enclose both objects while
the centre stayed where the keeper was first seen.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_services as osv  # noqa: E402


def obs(frame_id, stamp):
    return NS(frame_id=frame_id, stamp=stamp)


HULL = {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0, "z_min": 0.0, "z_max": 2.0}


def test_survivor_inherits_deduped_time_ordered_sightings():
    keeper = NS(centroid=[0.0, 0.0, 0.0], observations=[obs("f1", 1.0), obs("f3", 3.0)])
    # f3 is shared. The keeper and the discard detected in ONE frame is one view of one
    # object, so it must not be counted twice.
    discard = NS(centroid=[2.0, 2.0, 2.0], observations=[obs("f2", 2.0), obs("f3", 3.0)])

    osv._absorb_into_keeper(keeper, discard, HULL)

    assert [o.frame_id for o in keeper.observations] == ["f1", "f2", "f3"], \
        [o.frame_id for o in keeper.observations]
    assert [round(c, 3) for c in keeper.centroid] == [1.0, 1.0, 1.0], keeper.centroid


def test_the_cap_is_the_one_the_recorder_already_uses():
    cap = osv.MAX_OBSERVATIONS_PER_OBJECT
    keeper = NS(centroid=[0.0, 0.0, 0.0],
                observations=[obs(f"k{i}", float(i)) for i in range(cap)])
    discard = NS(centroid=None,
                 observations=[obs(f"d{i}", float(cap + i)) for i in range(10)])

    osv._absorb_into_keeper(keeper, discard, HULL)

    assert len(keeper.observations) == cap, len(keeper.observations)
    # The OLDEST are dropped, keeping the most recent views -- the rule
    # object_manager_6._record_sighting already applies, reading the same config key.
    assert keeper.observations[-1].frame_id == "d9", keeper.observations[-1].frame_id
    assert keeper.observations[0].frame_id == "k10", keeper.observations[0].frame_id


def test_a_discard_with_nothing_to_give_changes_only_the_centre():
    keeper = NS(centroid=[0.0, 0.0, 0.0], observations=[obs("f1", 1.0)])
    before = list(keeper.observations)

    osv._absorb_into_keeper(keeper, NS(centroid=None, observations=[]), HULL)

    assert keeper.observations == before
    assert [round(c, 3) for c in keeper.centroid] == [1.0, 1.0, 1.0], keeper.centroid


def test_a_malformed_hull_leaves_the_centre_alone_rather_than_blanking_it():
    # `_centroid_from_bbox` answers None on a box it cannot read. Overwriting a real centre
    # with None would be worse than keeping a stale one: every reader of `centroid` treats
    # None as "never seen" and `_record_sighting` returns early on it.
    keeper = NS(centroid=[9.0, 9.0, 9.0], observations=[])

    osv._absorb_into_keeper(keeper, NS(centroid=None, observations=[]), {"x_min": 0.0})

    assert keeper.centroid == [9.0, 9.0, 9.0], keeper.centroid


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"  ok  {name}")
    print(f"{n} checks passed")
