"""Capture-deduplicated evidence, independent of ontology admission and track IDs.

A detector score is recorded as a score, never interpreted as a calibrated probability.
Confirmation means repeated acquisition, not proof of existence or independent viewpoints.
"""
from collections import Counter
from copy import deepcopy
import math


def normalized(value):
    value = str(value or '').split('#')[0].strip().lower().replace('_', ' ')
    return '' if value in ('', 'unknown', 'none', 'null', 'n/a') else value


def record(obj, evidence):
    capture = str(evidence.get('capture_id') or '').strip()
    if not capture:
        return  # Missing provenance must not manufacture independent confirmations.
    history = getattr(obj, 'perception_evidence', {})
    entries = history.setdefault(capture, {})
    observation = str(evidence.get('observation_id') or capture)
    entries.setdefault(observation, deepcopy(evidence))
    obj.perception_evidence = history


def merge(keeper, discard):
    for entries in getattr(discard, 'perception_evidence', {}).values():
        for evidence in entries.values():
            record(keeper, evidence)


def summary(obj):
    history = getattr(obj, 'perception_evidence', {})
    scores = []
    for entries in history.values():
        valid = [e.get('boxer_confidence') for e in entries.values()]
        valid = [float(s) for s in valid if isinstance(s, (float, int)) and math.isfinite(s)]
        if valid:
            scores.append(max(valid))
    return dict(state=('confirmed' if len(history) >= 2 else 'tentative' if history else 'unmeasured'),
                capture_count=len(history), confidence_source='boxer_3d' if scores else None,
                confidence_min=min(scores) if scores else None,
                confidence_mean=sum(scores) / len(scores) if scores else None)


def stabilize(obj, scene_graph, objects=()):
    """Strict majority from >=2 captures; conflicts within one capture abstain.

    Keep the incumbent on ties. Unknown answers never erase known attributes.
    Identity stays object_id; update the legacy room label references on renaming.
    """
    history = getattr(obj, 'perception_evidence', {})
    old_label, old_description = obj.label, obj.description
    for field in ('label', 'color', 'material'):
        votes = Counter()
        for entries in history.values():
            values = {normalized(e.get(field)) for e in entries.values()}
            values.discard('')
            if len(values) == 1:
                votes.update(values)
        if not votes:
            continue
        winner, count = votes.most_common(1)[0]
        if count < 2 or count * 2 <= sum(votes.values()):
            continue
        if normalized(getattr(obj, field, '')) == winner:
            continue
        if field == 'label':
            suffix = '#' + obj.label.split('#', 1)[1] if '#' in obj.label else ''
            obj.label = winner + suffix
            # Ontology alignment belongs to the previous classification.
            obj.onto_aligned = False
        else:
            setattr(obj, field, winner)
    if obj.label != old_label:
        room_id = getattr(obj, 'room_id', None)
        room = scene_graph.get(room_id, {})
        labels = room.get('objects', [])
        if isinstance(labels, list) and old_label in labels:
            shared = any(other is not obj and other.label == old_label
                         and getattr(other, 'room_id', None) == room_id for other in objects)
            if not shared:
                labels[:] = [label for label in labels if label != old_label]
            if obj.label not in labels:
                labels.append(obj.label)
        # Pick an actual description supporting the winning class, not an invented blend.
        candidates = [e for entries in history.values() for e in entries.values()
                      if normalized(e.get('label')) == normalized(obj.label)
                      and normalized(e.get('description'))]
        if candidates:
            obj.description = min(candidates, key=lambda e: str(e.get('observation_id', '')))['description']
    return obj.description != old_description
