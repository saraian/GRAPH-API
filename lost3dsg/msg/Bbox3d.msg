string label
ObservationRef observation
float32 x_min
float32 x_max
float32 y_min
float32 y_max
float32 z_min
float32 z_max

# GA-100. The PCA-oriented box computed in detection_pipeline.py never left the perception
# node: this message carried only the axis-aligned extents, so `_publish_bbox_array`'s
# hasattr guard silently dropped yaw/oriented_center/oriented_extents on every detection.
# Anything diagonal -- a desk at an angle, a bed against a corner -- arrived at the object
# manager as the axis-aligned box the design says under-measures it.
#
# has_orientation is NOT redundant: without it a yaw of 0.0 from a real measurement is
# indistinguishable from the 0.0 a message gets when nothing was computed, which is the
# absent-reported-as-measured defect this review keeps finding.
bool has_orientation
float32 yaw
float32[3] oriented_center
float32[3] oriented_extents

# GA-186. The DETECTOR'S 2D box in the frame that produced this detection, in pixels
# (x_min, y_min, x_max, y_max). It never left the perception node, so the object manager
# could not tell "two detections of one object" from "two objects in one frame" -- and the
# co-visibility channel, which is a HARD NEGATIVE, had to abstain on every co-visible pair
# rather than assume. Measured on run 20260901_055513: 17 co-visible pairs shared a ground
# truth id, with 2D IoU from 0.482 to 0.997, and a veto that assumed disjointness would have
# refused every one of those merges.
#
# has_bbox_2d follows has_orientation above and for the same reason: without it, the (0,0,0,0)
# of a detection whose box was never carried is indistinguishable from a real box at the image
# origin, and a fabricated overlap of 0.0 is exactly what turns the abstention back into a
# wrong veto.
bool has_bbox_2d
float32[4] bbox_2d

# GA-190. The crop's APPEARANCE EMBEDDING for this detection, from the detector backend.
# It is computed by the backend (or by the local CLIP appearance encoder on the
# unified-VLM path) and is carried with the detection so it cannot be confused
# with a label-keyed sidecar or with the text/description embedding.
#
# The association layer's appearance channel needs exactly this, per VIEW, and had been
# falling back to a shape descriptor derived from the very box the objects already agree
# on. Carrying it here is what lets appearance be evidence rather than a restatement.
#
# has_clip_embedding for the third time on this message and the same reason as the other
# two: an all-zero vector from an absent embedding is a real embedding as far as any cosine
# is concerned, and the channel must ABSTAIN rather than score a fabricated one.
bool has_clip_embedding
float32[] clip_embedding

# Multi-observation geometry. The enclosing Bbox3dArray.cycle_id identifies
# this view. Keys are flattened XYZ int32 triplets in the map frame at
# fusion_voxel_size_m resolution. The explicit flag preserves absence across
# the typed boundary; an empty array is never treated as a measured cloud.
bool has_fusion_voxels
float32 fusion_voxel_size_m
int32[] fusion_voxel_keys
