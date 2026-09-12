string label
# Stable per-frame detector identity, shared with Bbox3d and detections.jsonl.
string detection_id
string description
string color 
string material
string shape
string crop_path
# W6. The route this description took: ok | model_abstained | parse_failed | call_failed |
# unanswered (nothing landed this cycle -- describer in flight or refused by the box check).
# Without it, a parse failure and a genuine model refusal are the same string downstream and
# the unknown-description rate conflates three different defects.
string status
# GA-108 (2026-09-07). WHERE THIS DESCRIPTION WAS TAKEN. The describer answers one or more
# cycles after the crop was submitted, and the per-frame ordinal `label` ("bed#2") names a
# DIFFERENT detection by then; run 20260906_223701 ended with 204 of 212 objects "unknown"
# although 496 descriptions landed. A late answer is published on /object_descriptions_late
# with the frame id and 2D box of the crop it describes, and the object manager hands it to
# the object whose sighting in that frame has that box. Empty origin_frame = same cycle.
string origin_frame
float32[4] origin_bbox_2d
