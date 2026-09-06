#!/usr/bin/env python3
"""Evidence-accumulating object association — the replacement for the gate-based merge.

WHY THIS EXISTS, in one paragraph. The shipped merge is a chain of thresholds over a single
scalar produced by `lost_similarity`. That scalar cannot express how much was measured, so a
pair with nothing comparable scores 1.0000 on label agreement alone and merges (GA-101),
while a pair that genuinely agreed on three axes can score lower. Raising the threshold
cannot help: the pairs it must catch sit ABOVE the ones with real evidence. And the pair the
whole exercise started from -- an air conditioner fragment 0.718 m from its whole, inside
every gate -- was never even OFFERED to the comparison, because candidate selection is
unlogged and unbounded in a way nobody could inspect.

THE DESIGN RULE, from the owner: work in probability space and ship NO FITTED CONSTANTS. A
real robot has no ground truth to fit against, and a constant tuned on one robot in one
building is a lie on the next. So every number here is one of three things, and the
docstring of each channel says WHICH:

  (a) a distributional quantile -- chi-squared at 3 degrees of freedom, universal;
  (b) a quantity MEASURED from this run -- the mapped volume, the number of rooms, an
      object's own observation spread;
  (c) a STATED DESIGN CHOICE with its rationale visible -- the false-merge/missed-merge
      cost ratio, which is a policy about which error hurts more, not a fact about a robot.

Ground truth is for VALIDATION ONLY. No GT number is read by this module at runtime.

THE STRUCTURE. Each channel returns a log-odds contribution in favour of "these two
detections are the same object", or None to ABSTAIN. Abstention is a first-class answer and
is never zero-by-accident: "no comparable view" must not read as "dissimilar", which is
GA-101's lesson carried into every new channel. Contributions sum across CHANNELS; across
FRAMES they replace rather than accumulate, because re-measuring unchanged geometry is not
new evidence (see Hypothesis -- my first version summed them and manufactured +47 log-odds
from one measurement repeated five times). A merge commits when the total crosses the
cost-ratio threshold AND has held for several consecutive updates, and it keeps enough
provenance to be undone.

NOT WIRED INTO THE SERVICE PATH. `_cb_merge_objects` is untouched and still runnable. This
module is pure Python plus numpy -- no ROS import -- so it can be exercised on a host with
no container, which is the only reason its self-check below can run at all.

Run `python3 association.py` for the self-check.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------
# Universal constants. Category (a): distributional quantiles, not tuned to anything.
# ---------------------------------------------------------------------------------------

# chi-squared 95th percentile at 3 degrees of freedom (3D position residual).
# A Mahalanobis distance-squared above this is outside the 95% ellipsoid.
CHI2_95_3DOF = 7.815

# chi-squared 99th percentile at 3 DoF, for the wider candidate-generation shell.
CHI2_99_3DOF = 11.345

# Log-odds are clipped to this magnitude per channel. Not a tuning knob: it stops a single
# channel's numerical blow-up (a near-zero null probability) from silently becoming a veto,
# which would defeat the point of accumulating evidence. A genuine veto is explicit.
MAX_CHANNEL_LOG_ODDS = 8.0


class Abstain:
    """Marker for 'this channel is not contributing a number'. Never a number.

    `measured` separates the two reasons a channel can decline, and they are NOT the same
    thing downstream (GA-186):

      measured=False -- THE EVIDENCE WAS NEVER COLLECTED. No observations, no frame ids, no
                        2D box. Nothing was looked at, so nothing can be concluded, and a
                        decision resting on another channel is unsupported.
      measured=True  -- THE EVIDENCE WAS COLLECTED AND IS INCONCLUSIVE FOR THIS CHANNEL.
                        Co-visibility on two overlapping detections is the case: the boxes
                        were compared and the pair is the duplicate-detection shape, which
                        is not evidence of TWO objects -- but it is very much not "we did
                        not look".

    Collapsing the two is what made a measured duplicate pair HOLD: `containment_unchecked`
    read the abstention as uncollected evidence and refused to commit a merge that the 2D
    overlap had just supported. A held duplicate is the exact failure association exists to
    fix, so the distinction is carried rather than inferred from the reason string.
    """

    __slots__ = ("reason", "measured")

    def __init__(self, reason, measured=False):
        self.reason = reason
        self.measured = measured

    def __repr__(self):
        return f"Abstain({self.reason!r}, measured={self.measured})"


VETO = "veto"  # a hard negative no accumulation may override


# ---------------------------------------------------------------------------------------
# Observation records. Channel 3 and channel 5 are both impossible without these.
# ---------------------------------------------------------------------------------------


class Observation:
    """One sighting of one object: which frame, from where, at what range.

    The tree records none of this today, which is exactly why co-visibility could not be
    used as a constraint and why appearance comparison had no way to know whether two
    descriptors were even comparable. Both channels below need it, so it is added here.
    """

    __slots__ = ("frame_id", "camera_position", "bearing", "range_m", "stamp", "centroid",
                 "bbox_2d", "appearance")

    def __init__(self, frame_id, camera_position, centroid, stamp=None, bbox_2d=None,
                 appearance=None):
        self.frame_id = frame_id
        self.camera_position = np.asarray(camera_position, dtype=float)
        self.centroid = np.asarray(centroid, dtype=float)
        self.stamp = stamp
        # GA-186: the DETECTOR's box in this frame, (x_min, y_min, x_max, y_max) in pixels,
        # or None when the frame carried none. None is what makes co-visibility abstain
        # instead of vetoing, so it must never be filled with a placeholder.
        self.bbox_2d = None if bbox_2d is None else [float(v) for v in bbox_2d]
        # GA-190: the crop's appearance embedding for THIS view, or None. Already computed
        # by the detector backend on a model it has loaded anyway, and previously written
        # to a sidecar nothing read. None stays None -- the appearance channel abstains on
        # a missing descriptor and must never see a zero vector standing in for one.
        self.appearance = (None if appearance is None
                           else np.asarray(appearance, dtype=float).ravel())
        d = self.centroid - self.camera_position
        self.range_m = float(np.linalg.norm(d))
        # Unit bearing from camera to object, in the map frame. Two observations are
        # "comparable" for appearance only when their bearings are close (see cone_ok).
        self.bearing = d / self.range_m if self.range_m > 1e-9 else np.array([1.0, 0.0, 0.0])


def shared_frame_overlap_2d(a, b):
    """-> the 2D IoU of two objects in a frame they SHARE, or None. GA-186.

    This is the function `AssocContext.overlap_2d_fn` wants. It answers the one question
    that decides whether co-visibility is a hard negative or an abstention: in a frame where
    both were detected, did the detector draw one region or two?

    Returns None -- and the channel then abstains -- whenever the answer cannot be measured:
    no shared frame, or either side's observation in that frame carried no 2D box. **A
    missing box is not a disjoint box.** Assuming disjointness is precisely the error that
    made this a wrong veto on 17 measured pairs (GA-181), so the absent case must not fall
    through to 0.0.

    MAX over shared frames, not mean: the pair is the same object if it was ever drawn as
    one region, and averaging a 0.99 overlap with three frames of partial occlusion would
    dilute exactly the evidence that matters.
    """
    if not a.observations or not b.observations:
        return None
    by_frame_b = {}
    for o in b.observations:
        if o.frame_id is not None and o.bbox_2d is not None:
            by_frame_b.setdefault(o.frame_id, []).append(o.bbox_2d)
    if not by_frame_b:
        return None
    best = None
    for oa in a.observations:
        if oa.frame_id is None or oa.bbox_2d is None:
            continue
        for box_b in by_frame_b.get(oa.frame_id, ()):
            iou = _iou_2d(oa.bbox_2d, box_b)
            if best is None or iou > best:
                best = iou
    return best


def _iou_2d(p, q):
    """Intersection over union of two (x_min, y_min, x_max, y_max) pixel boxes."""
    px1, py1, px2, py2 = min(p[0], p[2]), min(p[1], p[3]), max(p[0], p[2]), max(p[1], p[3])
    qx1, qy1, qx2, qy2 = min(q[0], q[2]), min(q[1], q[3]), max(q[0], q[2]), max(q[1], q[3])
    iw = max(0.0, min(px2, qx2) - max(px1, qx1))
    ih = max(0.0, min(py2, qy2) - max(py1, qy1))
    inter = iw * ih
    union = (px2 - px1) * (py2 - py1) + (qx2 - qx1) * (qy2 - qy1) - inter
    return 0.0 if union <= 0 else inter / union


def cone_ok(obs_a, obs_b, half_angle_rad):
    """Do these two sightings look at the object from close enough to the same direction?

    `half_angle_rad` is a STATED DESIGN CHOICE (category c) describing when two views of a
    non-symmetric object are expected to look alike. It is not fitted -- it encodes the
    assumption that appearance is roughly stable within a cone, and it is a config knob so
    the assumption is visible rather than buried.
    """
    c = float(np.clip(np.dot(obs_a.bearing, obs_b.bearing), -1.0, 1.0))
    return math.acos(c) <= half_angle_rad


# ---------------------------------------------------------------------------------------
# Per-object position covariance. Category (b): measured, never fitted.
# ---------------------------------------------------------------------------------------


def position_covariance(observations, depth_sparsity=0.0, pose_sigma_m=0.0):
    """Covariance of the object's position estimate, in metres^2.

    Grows with range, depth sparsity and pose uncertainty; shrinks with observation count.
    Every term is measured or supplied by the caller from what the sensor reported:

      * RANGE. Stereo/depth error grows with the square of range for a fixed disparity
        quantisation, so the per-observation variance uses range^2 as its scale. The
        coefficient is not a fitted constant: it is the relative depth error the sensor
        itself reports, passed in as `depth_sparsity` (0 = dense and trusted).
      * SPREAD. When an object has been seen more than once, the SAMPLE covariance of the
        observed centroids is a direct measurement of how much its position estimate moves.
        That is preferred over any model whenever it is available.
      * COUNT. Averaging n independent observations divides the covariance by n.

    With ONE observation there is no spread to measure, so the range term carries it alone.
    With none, this returns None -- an object with no observation record has no measurable
    uncertainty, and pretending otherwise is what a fitted constant would do.
    """
    if not observations:
        return None

    n = len(observations)
    centroids = np.array([o.centroid for o in observations], dtype=float)

    mean_range = float(np.mean([o.range_m for o in observations]))
    # Relative depth error -> absolute standard deviation at this range, plus the pose
    # uncertainty of the platform itself. Both are supplied by the caller from sensor and
    # localisation reports, not chosen here.
    sigma_range = mean_range * float(depth_sparsity) + float(pose_sigma_m)
    model_cov = np.eye(3) * max(sigma_range ** 2, 1e-9)

    if n >= 2:
        # Measured spread of the actual observations. np.cov wants variables in rows.
        sample_cov = np.cov(centroids.T, ddof=1)
        sample_cov = np.atleast_2d(sample_cov)
        # Take the elementwise maximum so a lucky run of near-identical detections cannot
        # claim more precision than the sensor model supports, and a genuinely scattered
        # object is not flattered by the model.
        cov = np.maximum(sample_cov, model_cov)
    else:
        cov = model_cov

    cov = cov / float(n)
    # Keep it symmetric positive-definite for the inversion below.
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-9
    return cov


def mahalanobis_sq(mu_a, cov_a, mu_b, cov_b):
    """Squared Mahalanobis distance between two position estimates under their joint covariance."""
    delta = np.asarray(mu_a, dtype=float) - np.asarray(mu_b, dtype=float)
    s = np.asarray(cov_a, dtype=float) + np.asarray(cov_b, dtype=float)
    return float(delta.T @ np.linalg.pinv(s) @ delta)


# ---------------------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------------------

_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _as_bounds(bbox):
    if bbox is None:
        return None
    try:
        return tuple(float(bbox[k]) for k in _KEYS)
    except (KeyError, TypeError):
        return None


def box_volume(b):
    return max(0.0, b[1] - b[0]) * max(0.0, b[3] - b[2]) * max(0.0, b[5] - b[4])


def box_centroid(b):
    return np.array([(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0])


def intersection_volume(a, b):
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    dz = min(a[5], b[5]) - max(a[4], b[4])
    if dx <= 0 or dy <= 0 or dz <= 0:
        return 0.0
    return dx * dy * dz


def hull_volume(a, b):
    """Volume of the smallest axis-aligned box containing both."""
    return ((max(a[1], b[1]) - min(a[0], b[0]))
            * (max(a[3], b[3]) - min(a[2], b[2]))
            * (max(a[5], b[5]) - min(a[4], b[4])))


def containment_ratio(a, b):
    """Intersection over the SMALLER volume. A fragment inside a whole scores ~1.0.

    This is the quantity plain IoU destroys. A 0.02 m^3 fragment sitting inside a 0.9 m^3
    air conditioner has IoU ~0.02 -- indistinguishable from two unrelated objects that
    barely touch -- and containment 1.0, which says exactly what is true: one of these is
    entirely inside the other.
    """
    inter = intersection_volume(a, b)
    smaller = min(box_volume(a), box_volume(b))
    return inter / smaller if smaller > 1e-12 else 0.0


def generalized_iou(a, b):
    """3D generalized IoU: IoU minus the fraction of the hull that neither box occupies.

    Unlike IoU it stays informative when the boxes do NOT overlap -- it goes negative, and
    more negative the further apart they are -- which is why it can be reported for every
    pair rather than only for touching ones.
    """
    inter = intersection_volume(a, b)
    union = box_volume(a) + box_volume(b) - inter
    hull = hull_volume(a, b)
    if hull <= 1e-12 or union <= 1e-12:
        return 0.0
    return inter / union - (hull - union) / hull


# ---------------------------------------------------------------------------------------
# CHANNEL 1 (PRIMARY) — containment / overlap
# ---------------------------------------------------------------------------------------


def channel_overlap(bounds_a, bounds_b, map_volume_m3):
    """Log-odds from 3D containment and generalized IoU. The channel that catches fragments.

    THE NULL, and why this has no fitted constant. Under "different objects", the two boxes
    are two independent objects in the mapped volume. The probability that an independent
    object's box would land so as to produce an intersection of volume V is approximately
    V / V_map -- the chance its centre falls in the overlap region. Under "same object" an
    overlap of that size is unremarkable, probability ~1. So

        log-odds  ~  log( 1 / (V_intersect / V_map) )  =  log(V_map / V_intersect)

    V_map is MEASURED from the map (category b) and V_intersect from the boxes. Nothing is
    tuned. The approximation is stated plainly: it treats object placement as uniform over
    the mapped volume, which is wrong in detail -- objects cluster on floors and against
    walls -- and wrong in the CONSERVATIVE direction, since real clustering makes chance
    overlap MORE likely and would lower this contribution.

    Containment scales it: an intersection that fills the smaller box is the fragment/whole
    case and gets the full weight; a glancing corner overlap gets proportionally less.

    Returns Abstain when either box is missing -- never 0.0, which would read as "measured,
    and they are unrelated".
    """
    if bounds_a is None or bounds_b is None:
        return Abstain("no bbox on one or both sides")
    if map_volume_m3 is None or map_volume_m3 <= 0:
        return Abstain("mapped volume unknown, no null to compare against")

    inter = intersection_volume(bounds_a, bounds_b)
    contain = containment_ratio(bounds_a, bounds_b)
    giou = generalized_iou(bounds_a, bounds_b)

    if inter <= 0.0:
        # No overlap at all. gIoU is still informative and negative; hand the separation
        # question to channel 2, which models it properly, and contribute only the mild
        # negative that "these boxes do not touch" deserves.
        return max(-MAX_CHANNEL_LOG_ODDS, giou), {
            "containment": 0.0, "giou": giou, "intersection_m3": 0.0}

    llr = math.log(map_volume_m3 / inter) * contain
    llr = float(np.clip(llr, -MAX_CHANNEL_LOG_ODDS, MAX_CHANNEL_LOG_ODDS))
    return llr, {"containment": contain, "giou": giou, "intersection_m3": inter}


# ---------------------------------------------------------------------------------------
# CHANNEL 2 — distributional separation, ONLY when overlap is zero
# ---------------------------------------------------------------------------------------


def channel_separation(mu_a, cov_a, mu_b, cov_b, map_volume_m3):
    """Log-odds from Mahalanobis separation. Expressed as log-odds, never as metres.

    Under "same object" the position difference is zero-mean Gaussian with covariance
    Sigma_a + Sigma_b, so its density at the observed delta is the multivariate normal
    density. Under "different objects" the delta is a draw from the mapped volume, density
    1 / V_map. The log-odds is the log ratio -- both terms are proper densities in the same
    3D position space, so the ratio is dimensionless and the metres cancel.

    That cancellation is the point. A raw distance in metres has no meaning without a scale;
    the same 0.7 m is decisive for a well-localised object seen forty times and meaningless
    for one seen once at eight metres range. The covariance supplies the scale, and it is
    measured (see position_covariance), not chosen.

    The 95% chi-squared gate at 3 DoF is reported alongside for inspection but is NOT applied
    here as a hard cut -- gating is candidate generation's job, and a channel that silently
    returned "impossible" would hide evidence the accumulator should see.
    """
    if cov_a is None or cov_b is None:
        return Abstain("no observation record, so no measurable position uncertainty")
    if map_volume_m3 is None or map_volume_m3 <= 0:
        return Abstain("mapped volume unknown, no null to compare against")

    d2 = mahalanobis_sq(mu_a, cov_a, mu_b, cov_b)
    s = np.asarray(cov_a) + np.asarray(cov_b)
    sign, logdet = np.linalg.slogdet(s)
    if sign <= 0:
        return Abstain("joint covariance not positive definite")

    # log N(delta; 0, S) = -0.5 * (d2 + logdet + k log 2pi), k = 3
    log_p_same = -0.5 * (d2 + logdet + 3.0 * math.log(2.0 * math.pi))
    log_p_diff = -math.log(map_volume_m3)
    llr = float(np.clip(log_p_same - log_p_diff, -MAX_CHANNEL_LOG_ODDS, MAX_CHANNEL_LOG_ODDS))
    return llr, {"mahalanobis_sq": d2, "chi2_95_gate": CHI2_95_3DOF,
                 "inside_95_ellipsoid": d2 <= CHI2_95_3DOF}


# ---------------------------------------------------------------------------------------
# CHANNEL 3 — co-visibility, a HARD NEGATIVE
# ---------------------------------------------------------------------------------------


def channel_covisibility(obs_a, obs_b, overlap_2d=None, duplicate_iou=0.30):
    """Two detections in one frame are different objects — UNLESS THEY ARE THE SAME PIXELS.

    GA-181, MEASURED AND IT OVERTURNS THE ORIGINAL PREMISE. I wrote that "a single frame
    cannot contain the same physical object twice as two separate detections". That is true
    of OBJECTS and false of DETECTIONS, and the distinction is the whole channel.

    Run 20260901_055513, 17 co-visible pairs that ground truth says are ONE object:

      towel radiator#1 | toilet#2          2D IoU 0.991   3D distance 0.000 m
      sink#1           | bathroom vanity#3 2D IoU 0.997   3D distance 0.000 m
      bathroom vanity#1| laundry basket#2  2D IoU 0.976   3D distance 0.000 m
      wastebasket#2    | wastebasket#3     2D IoU 0.488   3D distance 0.058 m
      mirror#2         | mirror#3          2D IoU 0.482   3D distance 0.230 m

    The high-IoU group is the SAME PIXELS detected twice under different open-vocabulary
    labels — the detector firing twice on one region. The GT join is not wrong; the boxes
    genuinely sit on one object. So a HARD veto here would refuse exactly the merges the
    system exists to make, every time it saw one.

    The rule therefore conditions on overlap: co-visible AND SPATIALLY DISJOINT is a hard
    negative, because two separated regions in one frame really are two objects. Co-visible
    and OVERLAPPING is not evidence either way — it is the duplicate-detection case, and the
    channel ABSTAINS rather than vetoing.

    `overlap_2d` is the pair's 2D IoU in the shared frame when the caller can supply it. With
    no overlap information the channel abstains rather than assuming disjointness, because
    assuming it is what made the veto wrong.

    THE ORIGINAL REASONING, still right for the DISJOINT case:

    This is the only channel that returns a veto rather than a number, and deliberately so.
    It is not evidence to be weighed -- it is a fact about the world: a single frame cannot
    contain the same physical object twice as two separate detections. Making it a large
    negative log-odds instead would let a sufficiently confident appearance match cancel it,
    which is precisely the failure this exists to prevent.

    Abstains, rather than returning "no veto", when either side has no observation record --
    absence of frame ids is not evidence that the two were never co-visible.
    """
    if not obs_a or not obs_b:
        return Abstain("no observation records to compare")
    frames_a = {o.frame_id for o in obs_a if o.frame_id is not None}
    frames_b = {o.frame_id for o in obs_b if o.frame_id is not None}
    if not frames_a or not frames_b:
        return Abstain("observations carry no frame ids")
    shared = frames_a & frames_b
    if not shared:
        return 0.0, {"covisible_frames": [], "n_covisible": 0}

    detail = {"covisible_frames": sorted(shared)[:8], "n_covisible": len(shared),
              "overlap_2d": overlap_2d}
    if overlap_2d is None:
        # No overlap information. ABSTAIN rather than veto: assuming disjointness is exactly
        # what made this channel wrong on 17 measured pairs.
        return Abstain("co-visible, but no 2D overlap available to rule out a duplicate "
                       "detection of one object")
    if overlap_2d >= duplicate_iou:
        # measured=True: the 2D boxes WERE compared and they overlap. Not evidence of two
        # objects, and equally not an absence of evidence -- see Abstain.
        return Abstain(f"co-visible but overlapping (2D IoU {overlap_2d:.2f}) — the "
                       f"duplicate-detection case, not evidence of two objects",
                       measured=True)
    return VETO, detail


# ---------------------------------------------------------------------------------------
# CHANNEL 4 — ontology, AFTER alignment only
# ---------------------------------------------------------------------------------------


def channel_ontology(type_a, type_b, aligned_a, aligned_b, disjoint_fn=None, n_types=None):
    """Prior or veto when BOTH sides are aligned to the ontology; abstain otherwise.

    An unaligned label is a string somebody's detector emitted. Treating "pillow" == "pillow"
    as ontological agreement is the same error as treating unknown == unknown as agreement:
    it converts absence of alignment into evidence. So this abstains unless both sides have
    actually been aligned.

    When both are aligned and the types are declared DISJOINT, that is a veto -- a chair is
    not a table, and no appearance score should be able to fuse them. When both are aligned
    to the same type, the contribution is log(n_types): the odds that two independently
    chosen objects share a type, measured from the ontology's own type count (category b),
    not chosen.
    """
    if not aligned_a or not aligned_b:
        return Abstain("one or both sides unaligned; alignment is required before use")
    if type_a is None or type_b is None:
        return Abstain("aligned but no type resolved")
    if disjoint_fn is not None and disjoint_fn(type_a, type_b):
        return VETO, {"type_a": type_a, "type_b": type_b, "disjoint": True}
    if type_a == type_b:
        if not n_types or n_types < 2:
            return Abstain("type count unknown, no null to compare against")
        return float(np.clip(math.log(n_types), 0.0, MAX_CHANNEL_LOG_ODDS)), {
            "type_a": type_a, "type_b": type_b, "n_types": n_types}
    return 0.0, {"type_a": type_a, "type_b": type_b, "disjoint": False}


# ---------------------------------------------------------------------------------------
# CHANNEL 5 — appearance re-ID, abstaining BY CONSTRUCTION
# ---------------------------------------------------------------------------------------


class ViewDescriptor:
    """A descriptor tagged with the bearing it was captured from.

    A single descriptor per object is what makes appearance comparison unreliable for
    anything that does not look the same from every side. A SET of view-tagged descriptors
    lets the comparison ask a question it can actually answer: is there a pair of views,
    one from each object, taken from close enough to the same direction to be comparable?
    """

    __slots__ = ("bearing", "vector", "kind")

    def __init__(self, bearing, vector, kind="extent"):
        b = np.asarray(bearing, dtype=float)
        n = np.linalg.norm(b)
        self.bearing = b / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        self.vector = np.asarray(vector, dtype=float)
        self.kind = kind


def extent_descriptor(bounds):
    """View-invariant shape descriptor: the two ratios of the sorted box extents.

    THE DEPLOYED BACKEND HAS NO CLIP PATH -- it is YOLO-World + SAM2 -- so there is no
    image embedding to compare, and inventing one would be a fitted model smuggled in as a
    feature. This is the view-invariant alternative the owner named: the sorted extents of
    the 3D box, reduced to two scale-free ratios, which describe the object's proportions
    rather than its size or its pose. A visual descriptor plugs into the same slot behind a
    flag when a backend provides one; see `score_pair(..., visual_descriptors=...)`.
    """
    if bounds is None:
        return None
    e = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
                 dtype=float)
    e = np.sort(np.abs(e))[::-1]
    if e[0] <= 1e-9:
        return None
    return np.array([e[1] / e[0], e[2] / e[0]])


def channel_appearance(descs_a, descs_b, cone_half_angle_rad, spread_a=None, spread_b=None):
    """MAX over comparable view pairs; ABSTAIN when no pair is comparable.

    GA-101's rule, carried into a new channel before it can repeat: "no comparable view"
    must never read as "dissimilar". If no pair of descriptors was taken from close enough
    to the same bearing, this abstains -- it does not return a low score.

    The scale for "how different is too different" is the objects' OWN measured descriptor
    spread (category b), not a constant. An object seen many times from many angles has a
    measurable spread in its descriptors; a difference well inside that spread is agreement.
    When neither side has enough observations to measure a spread, this abstains rather than
    fall back on a chosen sigma.
    """
    if not descs_a or not descs_b:
        return Abstain("no descriptors on one or both sides")

    # GA-190: same KIND, then same bearing cone. The shape guard below already skips a
    # 512-d appearance vector against a 2-d extent ratio, so mixed kinds were safe BY
    # ACCIDENT -- but two descriptor families that happened to share a length would have
    # been compared as if they measured the same thing, and the log-odds would have looked
    # like evidence. Kind is the layer the claim holds at; check it rather than rely on a
    # coincidence of dimensions.
    pairs = [(da, db) for da in descs_a for db in descs_b
             if da.kind == db.kind
             and cone_ok_vec(da.bearing, db.bearing, cone_half_angle_rad)]
    if not pairs:
        return Abstain("no pair of views of the same kind inside the comparability cone")

    sigma = _descriptor_sigma(spread_a, spread_b)
    if sigma is None:
        return Abstain("descriptor spread not measurable; no scale for the comparison")

    best = None
    for da, db in pairs:
        if da.vector.shape != db.vector.shape:
            continue
        d = float(np.linalg.norm(da.vector - db.vector))
        z = d / sigma
        # Same object: difference is noise, ~N(0, sigma) per component. Different objects:
        # the descriptor is spread over its own range, so a match this close is unlikely.
        # Ratio of a standard normal density to a diffuse one over the descriptor's range.
        llr = -0.5 * z * z + math.log(max(1e-6, 1.0 / sigma))
        best = llr if best is None else max(best, llr)
    if best is None:
        return Abstain("comparable views had incompatible descriptor shapes")
    return float(np.clip(best, -MAX_CHANNEL_LOG_ODDS, MAX_CHANNEL_LOG_ODDS)), {
        "comparable_pairs": len(pairs), "sigma": sigma}


def cone_ok_vec(bearing_a, bearing_b, half_angle_rad):
    c = float(np.clip(np.dot(bearing_a, bearing_b), -1.0, 1.0))
    return math.acos(c) <= half_angle_rad


def _descriptor_sigma(spread_a, spread_b):
    vals = [s for s in (spread_a, spread_b) if s is not None and s > 1e-9]
    if not vals:
        return None
    return float(max(vals))


# ---------------------------------------------------------------------------------------
# CHANNEL 6 — room as a PRIOR, never a veto
# ---------------------------------------------------------------------------------------


def channel_room(room_a, room_b, n_rooms):
    """Same room raises the odds; different rooms lower them; unknown ABSTAINS.

    Not a veto, because room assignment is itself an inference that can be wrong at a
    boundary, and vetoing on it would make one geometry error permanent. And unknown/unknown
    must not read as agreement -- that is the original sin this whole review began with.

    The magnitude is measured, not chosen: two independently placed objects share a room
    with probability about 1/n_rooms, so observing a shared room contributes log(n_rooms).
    n_rooms comes from the map (category b).
    """
    known_a = room_a is not None and str(room_a).strip().lower() not in ("", "unknown", "none")
    known_b = room_b is not None and str(room_b).strip().lower() not in ("", "unknown", "none")
    if not known_a or not known_b:
        return Abstain("room unknown on one or both sides")
    if not n_rooms or n_rooms < 2:
        return Abstain("fewer than two rooms mapped; no null to compare against")
    if room_a == room_b:
        return float(np.clip(math.log(n_rooms), 0.0, MAX_CHANNEL_LOG_ODDS)), {
            "room_a": room_a, "room_b": room_b, "same": True}
    return float(np.clip(-math.log(n_rooms), -MAX_CHANNEL_LOG_ODDS, 0.0)), {
        "room_a": room_a, "room_b": room_b, "same": False}


# ---------------------------------------------------------------------------------------
# Scoring a pair
# ---------------------------------------------------------------------------------------


class PairScore:
    """The full record of one comparison: total log-odds, every channel, every abstention.

    This object IS the diagnostic artefact. The old path recorded a float and lost the
    reason; a merge then destroyed the objects whose fields would have explained it. Here
    the per-channel breakdown and the abstentions are computed at comparison time and
    survive whatever happens to the objects afterwards.
    """

    __slots__ = ("total", "channels", "abstentions", "vetoed_by", "_measured_abstentions")

    def __init__(self):
        self.total = 0.0
        self.channels = {}
        self.abstentions = {}
        self.vetoed_by = []
        self._measured_abstentions = set()

    def add(self, name, result):
        if isinstance(result, Abstain):
            self.abstentions[name] = result.reason
            if result.measured:
                # The channel looked and could not conclude. Recorded separately from the
                # reason text so `containment_unchecked` reads a flag, not a string.
                self._measured_abstentions.add(name)
            return
        value, detail = result
        if value == VETO:
            self.vetoed_by.append(name)
            self.channels[name] = {"veto": True, **detail}
            return
        self.channels[name] = {"log_odds": value, **detail}
        self.total += value

    @property
    def vetoed(self):
        return bool(self.vetoed_by)

    @property
    def evidence_count(self):
        """How many channels actually measured something. Zero means: decide nothing.

        GA-310. This used to be `len(self.channels)`, and a channel that wrote 0.0 -- "consulted,
        likelihood ratio 1" -- is IN that dictionary. Measured over 437 rows of run 192014, all five
        merges had evidence_count >= 3 with only ONE channel non-zero, so `merge_min_evidence`, which
        exists to refuse a merge on overlap alone, could never refuse anything. A veto counts: it is
        the strongest measurement a channel can make. An Abstain never reaches `channels` at all, so
        the abstain / 0.0 / not-consulted distinction the Abstain class carries is untouched here.
        """
        return sum(1 for c in self.channels.values() if c.get("veto") or c.get("log_odds"))

    @property
    def containment_unchecked(self):
        """True when overlap is carrying the decision and NOTHING could contradict it.

        MEASURED, NOT ANTICIPATED. Replaying run 20260831_200858 through this module, 143 of
        153 replayable pairs agreed with what shipped and all 10 disagreements ran one way --
        refuse -> merge -- and they were pillows being merged INTO THE BED THEY REST ON:
        pillow#1 + bed at containment 0.850, bed + pillow#4 at 0.686, pillow#3 + bed at
        0.417. Two more were distinct pillow instances at containment 1.000.

        CONTAINMENT DOES NOT DISTINGUISH "A FRAGMENT OF X" FROM "AN OBJECT RESTING ON X".
        That is the whole weakness of making it primary, and the replay found it in the same
        scene where the shipped system fused a pillow with a bed at 0.954. A design that
        merges them MORE readily is worse, not better.

        The channel that separates the two cases is CO-VISIBILITY: a pillow and the bed under
        it appear in one frame, which is a hard negative no overlap can override. It was
        unavailable in the replay because the archive carries no observation records -- so
        the constraint that makes containment safe is exactly the one the data lacks.

        So when co-visibility could not be evaluated and overlap is carrying the decision,
        this reports that the conclusion is unsupported. The caller holds instead of
        committing. Abstain rather than assert: the evidence that would refute this merge was
        not merely absent, it was UNCOLLECTED.
        """
        if "covisibility" in self.channels:
            return False                      # it was evaluated; nothing to warn about
        if "covisibility" in self._measured_abstentions:
            # GA-186. It WAS evaluated: the 2D boxes were compared and the pair is the
            # duplicate-detection shape. That is the case containment is RIGHT about, so
            # holding here would refuse exactly the merges this module exists to make.
            # Only an UNCOLLECTED co-visibility leaves overlap unsupported.
            return False
        ov = self.channels.get("overlap", {}).get("log_odds")
        if ov is None or ov <= 0 or self.total <= 0:
            return False
        return ov >= 0.5 * self.total

    def as_record(self):
        return {
            "total_log_odds": None if self.vetoed else round(self.total, 4),
            "vetoed": self.vetoed,
            "vetoed_by": list(self.vetoed_by),
            "evidence_count": self.evidence_count,
            "channels": self.channels,
            "abstentions": self.abstentions,
        }


def score_pair(a, b, ctx):
    """Run every channel over one pair. `a` and `b` are AssocObject; `ctx` is AssocContext."""
    s = PairScore()

    bounds_a, bounds_b = _as_bounds(a.bbox), _as_bounds(b.bbox)

    # 3 first: a veto makes the rest moot for the decision, but the channels still run so
    # the record shows what the evidence WOULD have said. Diagnosis needs the disagreement.
    # GA-181: the 2D overlap decides whether co-visibility is a veto or an abstention. With
    # no 2D boxes to compare, `overlap_2d` stays None and the channel abstains rather than
    # assuming the two detections are disjoint.
    s.add("covisibility", channel_covisibility(a.observations, b.observations,
                                               overlap_2d=ctx.overlap_2d(a, b)))
    s.add("overlap", channel_overlap(bounds_a, bounds_b, ctx.map_volume_m3))

    ov = s.channels.get("overlap", {})
    if ov.get("intersection_m3", 0.0) <= 0.0:
        # Channel 2 runs ONLY when the boxes do not overlap, as specified: when they do,
        # containment already answers the question and separation would double-count it.
        s.add("separation", channel_separation(
            a.centroid, a.covariance, b.centroid, b.covariance, ctx.map_volume_m3))
    else:
        s.abstentions["separation"] = "boxes overlap; containment already answers this"

    s.add("ontology", channel_ontology(
        a.onto_type, b.onto_type, a.onto_aligned, b.onto_aligned,
        disjoint_fn=ctx.disjoint_fn, n_types=ctx.n_types))
    s.add("appearance", channel_appearance(
        a.descriptors, b.descriptors, ctx.cone_half_angle_rad,
        a.descriptor_spread, b.descriptor_spread))
    s.add("room", channel_room(a.room_id, b.room_id, ctx.n_rooms))
    return s


# ---------------------------------------------------------------------------------------
# Candidate generation — kNN with a COVARIANCE-DERIVED radius, and a log of what it excluded
# ---------------------------------------------------------------------------------------


def search_radius(obj, ctx):
    """How far this object should look for candidates, in metres.

    Derived FROM the covariance, not fixed. The radius is the 99% chi-squared shell of the
    object's own position uncertainty: sqrt(chi2_99_3dof * lambda_max(Sigma)), plus the
    object's own bounding-box reach so a large object can still meet a fragment at its edge.

    A FIXED radius is what would recreate the air-conditioner failure: a well-localised
    object and a badly-localised one need different search distances, and a single number
    chosen for the average case excludes exactly the uncertain pairs that most need looking
    at. An uncertain object searches wider, by construction.
    """
    reach = 0.0
    bounds = _as_bounds(obj.bbox)
    if bounds is not None:
        e = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        reach = float(np.linalg.norm(e) / 2.0)
    if obj.covariance is None:
        # No observation record -> no measurable uncertainty. Fall back to the object's own
        # extent and SAY SO in the exclusion log rather than invent a distance.
        return reach, "extent only (no covariance)"
    lam = float(np.max(np.linalg.eigvalsh(obj.covariance)))
    return reach + math.sqrt(CHI2_99_3DOF * max(lam, 0.0)), "covariance shell + extent"


def generate_candidates(objects, ctx, k=None):
    """-> (offered, excluded). BOTH are returned, because the exclusions are the diagnosis.

    Candidate selection is unlogged in the shipped system, which is why nobody can yet say
    why the air-conditioner pair -- 0.718 m apart and inside every gate -- was never
    compared. A pair that is never offered is invisible in exactly the way refusals were
    before they were logged. So this returns what it excluded, with the reason and the
    numbers, and the caller is expected to record it.
    """
    offered, excluded = [], []
    radii = {}
    for o in objects:
        radii[id(o)] = search_radius(o, ctx)

    for i in range(len(objects)):
        a = objects[i]
        r_a, basis_a = radii[id(a)]
        for j in range(i + 1, len(objects)):
            b = objects[j]
            r_b, basis_b = radii[id(b)]
            d = float(np.linalg.norm(np.asarray(a.centroid) - np.asarray(b.centroid)))
            reach = r_a + r_b
            if d <= reach:
                offered.append((a, b, {"distance_m": d, "reach_m": reach,
                                       "basis_a": basis_a, "basis_b": basis_b}))
            else:
                excluded.append((a, b, {"distance_m": d, "reach_m": reach,
                                        "reason": "outside the combined covariance shell",
                                        "basis_a": basis_a, "basis_b": basis_b}))

    if k is not None:
        # Keep the k nearest per object, but record what the cap removed -- a cap that
        # silently drops a pair is the same defect as an unlogged gate.
        per_obj = {}
        for a, b, meta in offered:
            per_obj.setdefault(id(a), []).append((meta["distance_m"], a, b, meta))
            per_obj.setdefault(id(b), []).append((meta["distance_m"], a, b, meta))
        keep = set()
        for _, lst in per_obj.items():
            lst.sort(key=lambda t: t[0])
            for _, a, b, _m in lst[:k]:
                keep.add((id(a), id(b)))
        kept, capped = [], []
        for a, b, meta in offered:
            if (id(a), id(b)) in keep:
                kept.append((a, b, meta))
            else:
                capped.append((a, b, {**meta, "reason": f"beyond the k={k} nearest"}))
        offered, excluded = kept, excluded + capped

    return offered, excluded


# ---------------------------------------------------------------------------------------
# Hypotheses — accumulate across frames, commit on the cost ratio, stay reversible
# ---------------------------------------------------------------------------------------


def commit_threshold(cost_ratio):
    """Log-odds at which a merge commits. A STATED DESIGN CHOICE (category c).

    Bayes decision theory: merge when the posterior odds of "same object" exceed the ratio
    of the cost of a false merge to the cost of a missed merge. With equal priors that is a
    log-odds threshold of log(cost_ratio).

    This is a POLICY, not a measurement, and it is the right place for the only judgement
    call in the module: D14's prefer-strict reading says a duplicate is visible and
    repairable while a wrong merge destroys an identity, so the cost ratio is well above 1.
    It is a config value so the choice is visible and arguable rather than buried in a
    threshold nobody can trace.
    """
    if cost_ratio is None or cost_ratio <= 0:
        raise ValueError("cost_ratio must be > 0; it is the false-merge/missed-merge cost ratio")
    return math.log(cost_ratio)


class Hypothesis:
    """A running belief that two tracks are one object.

    HOW THIS ACCUMULATES, AND A FLAW IN MY OWN FIRST DESIGN THAT THE SELF-CHECK EXPOSED.
    The obvious reading of "accumulate log-odds across frames" is to add each frame's score
    to a running total. That is WRONG here and the self-check caught it: re-scoring the same
    two static boxes in five consecutive frames took the total from +9.4 to +47.1 without a
    single new measurement. Geometry that has not changed is not new evidence, and summing
    it manufactures certainty out of repetition -- the same error as GA-101 in a new place,
    where a number grows without anything being measured.

    So the channels are functions of ACCUMULATED STATE, and the total is RECOMPUTED from
    the current state rather than summed over time. Evidence still grows -- but through the
    state: more observations tighten the covariance, more views make appearance comparable,
    a new detection changes the boxes. That growth is real; repetition is not.

    "One lucky frame must not commit a merge" is then enforced by PERSISTENCE instead of
    summation: the decision must hold for `min_consecutive` successive updates before it
    commits. A transient geometry error has to survive being re-measured.

    The full history is retained regardless, which is what makes the merge reversible and
    retires "the merge destroys its own diagnostic evidence".
    """

    __slots__ = ("key", "state", "history", "vetoed_by", "committed", "_streak")

    def __init__(self, key):
        self.key = key
        self.state = {}
        self.history = []
        self.vetoed_by = []
        self.committed = False
        self._streak = 0

    @property
    def total(self):
        if self.vetoed_by:
            return float("-inf")
        return float(sum(self.state.values()))

    def update(self, pair_score, frame_id=None):
        if pair_score.vetoed:
            # A veto is permanent for this pair: co-visibility and ontological disjointness
            # are facts about the world, not evidence that later evidence can outweigh.
            for name in pair_score.vetoed_by:
                if name not in self.vetoed_by:
                    self.vetoed_by.append(name)
            self._streak = 0
            self.history.append({"frame": frame_id, "veto": list(pair_score.vetoed_by)})
            return self
        # REPLACE, never accumulate: each channel reports on the current state.
        for name, ch in pair_score.channels.items():
            if "log_odds" in ch:
                self.state[name] = ch["log_odds"]
        self.history.append({"frame": frame_id,
                             "total": round(self.total, 4),
                             "containment_unchecked": pair_score.containment_unchecked,
                             "evidence_count": pair_score.evidence_count,
                             "channels": {k: v.get("log_odds")
                                          for k, v in pair_score.channels.items()},
                             "abstentions": dict(pair_score.abstentions)})
        return self

    def decide(self, threshold, min_evidence=1, min_consecutive=1):
        """-> (decision, reason). Never merges on zero measured evidence."""
        if self.vetoed_by:
            return "reject", f"vetoed by {', '.join(self.vetoed_by)}"
        measured = [h for h in self.history if h.get("evidence_count", 0) >= min_evidence]
        if not measured:
            # GA-101 one level up: a total built from nothing measured is not a confident
            # answer, it is the absence of one.
            return "abstain", "no update contributed a measured channel"
        last = self.history[-1] if self.history else {}
        if last.get("containment_unchecked") and self.total >= threshold:
            self._streak = 0
            return "hold", ("containment carries the decision and co-visibility was never "
                            "recorded — the hard negative that would refute it is uncollected")
        if self.total >= threshold:
            self._streak += 1
            if self._streak >= min_consecutive:
                return "merge", (f"log-odds {self.total:.3f} >= {threshold:.3f} "
                                 f"for {self._streak} consecutive updates")
            return "hold", (f"log-odds {self.total:.3f} >= {threshold:.3f} but only "
                            f"{self._streak}/{min_consecutive} consecutive")
        self._streak = 0
        return "hold", f"log-odds {self.total:.3f} < {threshold:.3f}"

    def provenance(self):
        """Everything needed to undo the merge and to explain it afterwards."""
        return {"key": list(self.key),
                "total_log_odds": (None if self.vetoed_by else round(self.total, 4)),
                "vetoed_by": list(self.vetoed_by),
                "state": {k: round(v, 4) for k, v in self.state.items()},
                "frames": list(self.history)}


# ---------------------------------------------------------------------------------------
# Plain containers so this module needs no ROS and no world_model import
# ---------------------------------------------------------------------------------------


class AssocObject:
    __slots__ = ("object_id", "label", "bbox", "centroid", "observations", "covariance",
                 "descriptors", "descriptor_spread", "room_id", "onto_type", "onto_aligned")

    def __init__(self, object_id, bbox=None, centroid=None, label=None, observations=None,
                 room_id=None, onto_type=None, onto_aligned=False,
                 descriptors=None, descriptor_spread=None,
                 depth_sparsity=0.0, pose_sigma_m=0.0):
        self.object_id = object_id
        self.label = label
        self.bbox = bbox
        self.observations = list(observations or [])
        b = _as_bounds(bbox)
        if centroid is not None:
            self.centroid = np.asarray(centroid, dtype=float)
        elif b is not None:
            self.centroid = box_centroid(b)
        elif self.observations:
            self.centroid = np.mean([o.centroid for o in self.observations], axis=0)
        else:
            self.centroid = np.zeros(3)
        self.covariance = position_covariance(self.observations, depth_sparsity, pose_sigma_m)
        self.room_id = room_id
        self.onto_type = onto_type
        self.onto_aligned = onto_aligned
        if descriptors is None and self.observations:
            # GA-190. A MEASURED appearance descriptor beats a derived shape one, so views
            # that carry an embedding form the set. The two are NOT mixed: an extent
            # descriptor is a function of the box this object already agrees on, so pooling
            # it with appearance would let shape similarity masquerade as visual similarity
            # in the same channel. Extent remains the fallback when no view was embedded.
            visual = [ViewDescriptor(o.bearing, o.appearance, kind="visual")
                      for o in self.observations if o.appearance is not None
                      and o.appearance.size]
            if visual:
                descriptors = visual
            elif b is not None:
                v = extent_descriptor(b)
                descriptors = ([ViewDescriptor(o.bearing, v) for o in self.observations]
                               if v is not None else [])
        self.descriptors = list(descriptors or [])
        self.descriptor_spread = descriptor_spread
        if self.descriptor_spread is None and len(self.descriptors) >= 2:
            vs = np.array([d.vector for d in self.descriptors])
            self.descriptor_spread = float(np.mean(np.std(vs, axis=0)))


class AssocContext:
    __slots__ = ("map_volume_m3", "n_rooms", "n_types", "cone_half_angle_rad",
                 "disjoint_fn", "cost_ratio", "overlap_2d_fn")

    def __init__(self, map_volume_m3=None, n_rooms=None, n_types=None,
                 cone_half_angle_rad=math.radians(45.0), disjoint_fn=None, cost_ratio=20.0,
                 overlap_2d_fn=None):
        self.map_volume_m3 = map_volume_m3
        self.n_rooms = n_rooms
        self.n_types = n_types
        self.cone_half_angle_rad = cone_half_angle_rad
        self.disjoint_fn = disjoint_fn
        self.cost_ratio = cost_ratio
        # GA-181: supplied by the caller, which is the only place 2D boxes in a shared frame
        # are available. None -> co-visibility abstains instead of vetoing.
        self.overlap_2d_fn = overlap_2d_fn

    def overlap_2d(self, a, b):
        return None if self.overlap_2d_fn is None else self.overlap_2d_fn(a, b)


# ---------------------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------------------

def _obs(frame, cam, centroid):
    return Observation(frame, cam, centroid)


def _box(cx, cy, cz, sx, sy, sz):
    return {"x_min": cx - sx / 2, "x_max": cx + sx / 2,
            "y_min": cy - sy / 2, "y_max": cy + sy / 2,
            "z_min": cz - sz / 2, "z_max": cz + sz / 2}


def demo():
    ctx = AssocContext(map_volume_m3=300.0, n_rooms=6, n_types=40, cost_ratio=20.0)
    thr = commit_threshold(ctx.cost_ratio)

    # --- 1. the air-conditioner shape: a fragment inside a whole -------------------------
    whole = AssocObject("ac_whole", bbox=_box(0, 0, 2.0, 0.9, 0.4, 0.4), room_id="bedroom",
                        observations=[_obs(1, [3, 0, 1], [0, 0, 2.0]),
                                      _obs(2, [3.2, 0.4, 1], [0.02, 0.01, 2.0])])
    frag = AssocObject("ac_frag", bbox=_box(0.25, 0, 2.0, 0.18, 0.3, 0.3), room_id="bedroom",
                       observations=[_obs(3, [3, 0, 1], [0.25, 0, 2.0])])
    s = score_pair(whole, frag, ctx)
    assert not s.vetoed
    assert s.channels["overlap"]["containment"] > 0.9, s.channels["overlap"]
    assert s.total > thr, (s.total, thr)
    print(f"  fragment inside whole : log-odds {s.total:+.2f} (commit at {thr:+.2f}) -> MERGE")
    print(f"      containment {s.channels['overlap']['containment']:.3f}, "
          f"gIoU {s.channels['overlap']['giou']:+.3f}")

    # --- 2. co-visibility vetoes, and no amount of similarity overrides it ---------------
    twin_a = AssocObject("p1", bbox=_box(0, 0, 0.5, 0.4, 0.4, 0.2), room_id="bedroom",
                         observations=[_obs(7, [2, 0, 1], [0, 0, 0.5])])
    twin_b = AssocObject("p2", bbox=_box(0.05, 0, 0.5, 0.4, 0.4, 0.2), room_id="bedroom",
                         observations=[_obs(7, [2, 0, 1], [0.05, 0, 0.5])])
    # GA-181: co-visible AND SPATIALLY DISJOINT is the hard negative.
    ctx_disjoint = AssocContext(map_volume_m3=300.0, n_rooms=6, n_types=40, cost_ratio=20.0,
                                overlap_2d_fn=lambda a, b: 0.0)
    s2 = score_pair(twin_a, twin_b, ctx_disjoint)
    assert s2.vetoed and "covisibility" in s2.vetoed_by
    # ...but co-visible and OVERLAPPING is the duplicate-detection case and must ABSTAIN.
    # Measured, run 20260901_055513: 17 co-visible pairs that GT says are ONE object, e.g.
    # "sink#1 | bathroom vanity#3" at 2D IoU 0.997 and 3D distance 0.000 m. A hard veto here
    # refuses exactly the merges the system exists to make.
    ctx_dup = AssocContext(map_volume_m3=300.0, n_rooms=6, n_types=40, cost_ratio=20.0,
                           overlap_2d_fn=lambda a, b: 0.997)
    s2b = score_pair(twin_a, twin_b, ctx_dup)
    assert not s2b.vetoed, "an overlapping co-visible pair must not be vetoed"
    assert "covisibility" in s2b.abstentions
    print(f"  co-visible + OVERLAPPING (IoU 0.997) -> abstains, not vetoed "
          f"({s2b.abstentions['covisibility'][:46]}...)")
    # and with no overlap information it abstains rather than assuming disjointness
    s2c = score_pair(twin_a, twin_b, ctx)
    assert not s2c.vetoed and "covisibility" in s2c.abstentions
    print("  co-visible + NO overlap info -> abstains (assuming disjoint is what was wrong)")

    h = Hypothesis(("p1", "p2")).update(s2, frame_id=7)
    for extra in range(3):
        h.update(s2, frame_id=8 + extra)
    assert h.decide(thr)[0] == "reject", h.decide(thr)
    print(f"  two detections, one frame : VETO ({s2.channels['covisibility']['n_covisible']} "
          f"shared frame) -> {h.decide(thr)[1]}")

    # --- 3. GA-101's rule in the new channels: nothing measurable => ABSTAIN, not 1.0 -----
    bare_a = AssocObject("b1", bbox=None, centroid=[0, 0, 0])
    bare_b = AssocObject("b2", bbox=None, centroid=[0.1, 0, 0])
    s3 = score_pair(bare_a, bare_b, ctx)
    assert s3.evidence_count == 0, s3.as_record()
    assert "overlap" in s3.abstentions and "separation" in s3.abstentions
    assert "appearance" in s3.abstentions and "room" in s3.abstentions
    h3 = Hypothesis(("b1", "b2")).update(s3, frame_id=1)
    assert h3.decide(thr)[0] == "abstain", h3.decide(thr)
    print(f"  nothing measurable : evidence_count 0, {len(s3.abstentions)} abstentions "
          f"-> {h3.decide(thr)[0].upper()} (never 1.0)")

    # --- 4. the search radius WIDENS with uncertainty -------------------------------------
    tight = AssocObject("t", bbox=_box(0, 0, 0, 0.2, 0.2, 0.2),
                        observations=[_obs(i, [1, 0, 0], [0, 0, 0]) for i in range(1, 9)],
                        depth_sparsity=0.01)
    loose = AssocObject("l", bbox=_box(0, 0, 0, 0.2, 0.2, 0.2),
                        observations=[_obs(1, [9, 0, 0], [0, 0, 0])],
                        depth_sparsity=0.08, pose_sigma_m=0.15)
    r_tight, _ = search_radius(tight, ctx)
    r_loose, _ = search_radius(loose, ctx)
    assert r_loose > r_tight, (r_tight, r_loose)
    print(f"  search radius : well-localised {r_tight:.3f} m < uncertain {r_loose:.3f} m "
          f"(fixed radius would exclude the uncertain pair)")

    # --- 5. selection reports what it EXCLUDED --------------------------------------------
    far = AssocObject("far", bbox=_box(20, 0, 0, 0.2, 0.2, 0.2),
                      observations=[_obs(1, [21, 0, 0], [20, 0, 0])], depth_sparsity=0.01)
    offered, excluded = generate_candidates([tight, loose, far], ctx)
    assert any(e[2]["reason"].startswith("outside") for e in excluded), excluded
    print(f"  candidate selection : {len(offered)} offered, {len(excluded)} excluded WITH "
          f"reasons -- the stage that is unlogged today")

    # --- 6. accumulation: weak-but-repeated crosses; one lucky frame does not -------------
    weak = AssocObject("w1", bbox=_box(0, 0, 0, 0.30, 0.30, 0.30), room_id="kitchen",
                       observations=[_obs(1, [2, 0, 0], [0, 0, 0])])
    weak2 = AssocObject("w2", bbox=_box(0.06, 0, 0, 0.30, 0.30, 0.30), room_id="kitchen",
                        observations=[_obs(2, [2, 0, 0], [0.06, 0, 0])])
    sw = score_pair(weak, weak2, ctx)
    # THE REGRESSION TEST FOR MY OWN BUG: re-scoring identical state must NOT grow the
    # total. The first version of Hypothesis summed per frame and turned one measurement
    # repeated five times into +47 log-odds of false confidence.
    many = Hypothesis(("w1", "w2"))
    totals = []
    for f in range(1, 6):
        many.update(sw, frame_id=f)
        totals.append(round(many.total, 6))
    assert len(set(totals)) == 1, f"unchanged state must not accumulate: {totals}"
    assert len(many.provenance()["frames"]) == 5, "but every update is still recorded"

    # Persistence, not summation, is what stops one lucky frame committing a merge.
    one = Hypothesis(("w1", "w2")).update(sw, frame_id=1)
    d_one = one.decide(thr, min_consecutive=3)
    persistent = Hypothesis(("w1", "w2"))
    for f in range(1, 4):
        persistent.update(sw, frame_id=f)
        d_persist = persistent.decide(thr, min_consecutive=3)
    print(f"  no double-counting : same state x5 -> total stays {totals[0]:+.2f} "
          f"(was +47.11 when summed)")
    print(f"  persistence : 1 update -> {d_one[0]}; 3 consecutive -> {d_persist[0]}")
    assert d_one[0] == "hold" and d_persist[0] == "merge", (d_one, d_persist)

    # --- 7. reversibility: the record outlives the objects --------------------------------
    prov = many.provenance()
    assert prov["key"] == ["w1", "w2"] and prov["frames"][0]["channels"] and prov["state"]
    print(f"  provenance : {len(prov['frames'])} frames retained with per-channel log-odds "
          f"-> the merge is reversible and explainable")

    # --- 8. room is a PRIOR, not a veto ---------------------------------------------------
    r_same = channel_room("kitchen", "kitchen", 6)[0]
    r_diff = channel_room("kitchen", "bedroom", 6)[0]
    assert r_same > 0 > r_diff
    assert isinstance(channel_room("unknown", "unknown", 6), Abstain)
    print(f"  room : same {r_same:+.2f}, different {r_diff:+.2f}, "
          f"unknown/unknown ABSTAINS (never agreement)")

    # --- 9. no fitted constants: every number traces to a quantile, a measurement or policy
    assert commit_threshold(20.0) == math.log(20.0)
    assert CHI2_95_3DOF == 7.815
    print(f"  constants : chi2 95%@3dof {CHI2_95_3DOF}, commit threshold "
          f"log(cost_ratio={ctx.cost_ratio:g}) = {thr:.3f}")

    print("\nassociation self-check OK")


if __name__ == "__main__":
    demo()
