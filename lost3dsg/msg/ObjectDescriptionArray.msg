std_msgs/Header header
# GA: THE CYCLE ID IS THE JOIN KEY, not the header stamp. object_manager_6 pairs a box
# array with its description array; it used to match on the exact (sec, nanosec) of the
# header. That held while both were produced inside one synchronous cycle. With the frame
# queue the box is published as soon as geometry is computed and the description waits on
# a VLM round trip (2.4 s median, 7.9 s p95 online), so the two streams drift: MEASURED on
# 20260911_193627, boxes were two minutes ahead of descriptions and the keys never met --
# 50 detections, 0 admissions, an overlay full of boxes and an empty object table.
# A stamp is provenance. This is identity, and it is minted once per cycle.
string cycle_id
ObjectDescription[] descriptions