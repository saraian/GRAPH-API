string label
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
