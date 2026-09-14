# Version 0 denotes a legacy reference without exact observation identity.
uint16 schema_version
string run_id
string producer_id
string capture_id
# exact_source | stamp_within_producer | unavailable
string capture_identity_kind
builtin_interfaces/Time capture_stamp
string camera_frame_id
string cycle_id
uint32 detection_index
string observation_id
