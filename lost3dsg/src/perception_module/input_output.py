import json
import os
import re
from datetime import datetime

import cv2
from std_msgs.msg import Header
from lost3dsg.msg import Bbox3dArray, ObjectDescriptionArray

from config import CFG, world_frame
from crop_context import build_context_crop
from perception_utils import compute_fov_volume_from_depth
from utils import draw_detections

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    _MODULE_DIR.split('/install/', 1)[0]
    if '/install/' in _MODULE_DIR
    else os.path.abspath(os.path.join(_MODULE_DIR, "../.."))
)


def resolve_output_root():
    """H12. ONE output root for EVERY writer in this module.

    `prepare_crops` wrote under `GRAPH_API_OUTPUT_DIR` (the run bundle) while
    `save_visualizations` and `write_perceptions_json` wrote under `PROJECT_ROOT/output`
    (the module path) — the two coincide only via the container's mapping, so with the env
    set anywhere else the visualizations and the per-cycle JSON orphaned while the crops
    landed in the bundle. One resolver: bundle env var if set, else the module path.
    """
    out_root = os.environ.get("GRAPH_API_OUTPUT_DIR")
    return out_root if out_root else os.path.join(PROJECT_ROOT, "output")


class PerceptionIOMixin:
    def make_header_msg(self, msg_type, stamp=None, frame_id=None):
        msg = msg_type()
        msg.header = Header(
            stamp=stamp if stamp is not None else self.get_clock().now().to_msg(),
            frame_id=world_frame() if frame_id is None else frame_id,
        )
        return msg

    def save_crop_file(self, path, image):
        try:
            cv2.imwrite(path, image)
        except Exception as exc:
            self.log_both("error", f"Background crop save failed ({path}): {exc}")

    def write_perceptions_json(self, perceptions_snapshot):
        # H12: the same root prepare_crops writes under — the bundle, when set.
        perceptions_path = os.path.join(resolve_output_root(), "actual_perceptions.json")
        try:
            os.makedirs(os.path.dirname(perceptions_path), exist_ok=True)
            with open(perceptions_path, "w") as file_obj:
                json.dump(perceptions_snapshot, file_obj, indent=4)
        except Exception as exc:
            self.log_both("error", f"Background JSON dump failed: {exc}")

    def publish_empty_state(self, depth, camera_info, cycle_stamp=None, fov_volume=None):
        stamp = cycle_stamp if cycle_stamp is not None else self.get_clock().now().to_msg()
        # The object cloud (/pcl_objects, latched) is deliberately NOT wiped here: an
        # empty cycle used to overwrite the last detection's cloud with a zero-point
        # message, so rviz showed object clouds only for the instant between two cycles.
        self.pub_object_descriptions.publish(self.make_header_msg(
            ObjectDescriptionArray, stamp=stamp, frame_id=world_frame()))

        empty_bboxes = self.make_header_msg(Bbox3dArray, stamp=stamp, frame_id=world_frame())
        # LAT-2. The caller has already computed this for THIS cycle, deliberately early
        # while the stamp is still inside the TF buffer. Recomputing it here repeated the
        # whole depth-to-map projection on every empty cycle for a value already in hand.
        # Recomputed only when a caller supplies nothing, so the function stays usable on
        # its own.
        fov = fov_volume if fov_volume is not None else compute_fov_volume_from_depth(
            depth, camera_info, self)
        if fov:
            for key, value in fov.items():
                setattr(empty_bboxes, f"fov_{key}", value)
        self.bbox_pub.publish(empty_bboxes)
        self.waiting_for_input = False

    def save_visualizations(self, image_raw, depth, detections):
        # H12: the same root prepare_crops writes under — the bundle, when set. The
        # `project_root` parameter is gone: the only caller passed PROJECT_ROOT, which is
        # exactly the module-path arm the resolver falls back to anyway.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_dir = os.path.join(resolve_output_root(), "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)

        cv2.imwrite(os.path.join(visualization_dir, f"bbox_{timestamp}.jpg"), draw_detections(image_raw.copy(), detections))

        depth_norm = cv2.applyColorMap(
            cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"),
            cv2.COLORMAP_JET,
        )
        depth_dir = os.path.join(visualization_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        cv2.imwrite(os.path.join(depth_dir, f"depth_{timestamp}.jpg"), depth_norm)

    def prepare_crops(self, detections, image_raw, frame_key):
        """`frame_key` (W2) and the output root (H12) are required, not optional: a crop
        without provenance is the exact misattribution W2 exists to stop, and a
        project_root parameter nobody read was the H12 split itself."""
        height, width = image_raw.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # GA-238. THE CROPS WERE WRITTEN INTO THE CONTAINER'S OWN SOURCE TREE.
        # `project_root` resolves inside /ws, which does not survive the container, so every
        # crop this function produced was discarded when the run ended. The bundle's crops
        # directory has been empty in every run, and the viewer shows its "no crop captured
        # yet" placeholder on every object card as the direct consequence.
        #
        # It is a RETENTION fault, not a perception one: `crop_meta` is populated on all
        # 1,530 detection rows of 20260901_174810_hm3d_00861, so the crops were constructed
        # correctly and then thrown away.
        #
        # `cropped_images` is the name the READER wants -- graph_api_bridge._crop_dirs
        # searches `_active_output_dir() / "cropped_images"` first. live_run.sh separately
        # creates `$RUN_DIR/crops`, a THIRD name that nothing reads or writes.
        # H12: one resolver for crops, visualizations and the per-cycle JSON alike.
        crops_dir = os.path.join(resolve_output_root(), "cropped_images")
        os.makedirs(crops_dir, exist_ok=True)

        crops = []
        for idx, det in enumerate(detections):
            x0 = max(0, min(int(det.bbox[0]), width - 1))
            y0 = max(0, min(int(det.bbox[1]), height - 1))
            x1 = max(0, min(int(det.bbox[2]), width))
            y1 = max(0, min(int(det.bbox[3]), height))
            x1 = max(x0 + 1, x1)
            y1 = max(y0 + 1, y1)

            # GA-108: ONE construction, selected by config, handed to BOTH consumers.
            # `tight` is the default and is byte-identical to the slice that used to be
            # written here -- asserted in crop_context's self-check, not assumed. The other
            # arms widen the window and outline the referent; the arm is a setting, never a
            # second implementation, so any difference between arms is the arm and not the
            # code path.
            crop, crop_meta = build_context_crop(
                image_raw, getattr(det, "mask", None),
                detector_box=(x0, y0, x1, y1),
                size=int(CFG["crop"]["describer_size"]),
                construction=str(CFG["crop"]["construction"]))
            if crop is None or crop.size == 0:
                # UNANSWERABLE, and recorded as such by crop_meta -- an image we could not
                # build is not the describer refusing (GA-108 decision 6).
                self.get_logger().warn(
                    f"No usable crop for {det.instance_label}: "
                    f"{crop_meta.get('status')} ({crop_meta.get('reason', '-')})")
                crops.append(None)
                continue

            # W7. The file gets the SAME pixels the describer sees. The old code drew a
            # 2 px green rectangle into the on-disk copy only: the admission gate
            # (GA-277) reads `crop_path`, the describer reads `['cropped']`, so the two
            # VLM judgments were made on different pixels and could never be compared --
            # and a full-frame green border is an artefact the gate model was never
            # meant to see. The referent is already marked by the mask contour
            # (crop_context decision 3); drawing the crop boundary added nothing.
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", det.instance_label)
            crop_path = os.path.join(crops_dir, f"crop_{safe_label}_{timestamp}_{idx}.jpg")
            self._io_executor.submit(self.save_crop_file, crop_path, crop.copy())
            crops.append({"cropped": crop, "label": det.instance_label, "idx": idx,
                          "crop_meta": crop_meta,
                          # W2. The frame and the 2D box this crop was taken from, so a
                          # deferred VLM result can be checked against the detection that
                          # receives it: instance_label is a per-frame ordinal ("chair#1")
                          # reused every cycle, and a result computed for one object must
                          # not attach to another that answers to the same string later.
                          # Required fields, not optional: a crop without provenance is
                          # exactly the misattribution this exists to stop.
                          "bbox": (x0, y0, x1, y1), "frame": frame_key,
                          # GA-277. The path is already computed for the disk write; carrying
                          # it is free and it is the ONLY way the admission gate can ever see
                          # the image. The gate receives geometry, not pixels, so a VLM check
                          # on a held decision had nothing to look at without this.
                          "path": crop_path})
        return crops

    # LAT-5. `publish_crops` DELETED, with its `/cropped_image` publisher. It encoded and
    # published every crop of every cycle to a topic NOTHING SUBSCRIBED TO: every crop
    # consumer in either tree reads the `cropped_images` DIRECTORY instead (the bridge's
    # `_crop_dirs`, the dashboard server, `test_graph_data.py`). Checked across both trees
    # before deleting; the only other references were the two dead snapshots.
