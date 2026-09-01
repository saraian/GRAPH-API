string label
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
