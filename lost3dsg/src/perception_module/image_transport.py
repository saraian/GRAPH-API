"""Small ROS-image metadata helper shared by the live bridge and its tests.

The helper deliberately does not import ROS or OpenCV. A ROS ``Image`` is a byte
buffer with a row stride; treating it as a tightly packed ``height * width * 3``
array drops padded physical-camera frames and can make a healthy
``/image_with_bb`` publisher look silent to the dashboard.
"""

import numpy as np


_SUPPORTED = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}


def ros_image_to_array(msg):
    """Return ``(H, W, C)`` uint8 pixels and the normalised encoding.

    ``step`` is bytes per row and may be larger than the visible row width. The
    alpha channel is retained here; the caller decides whether and how to convert
    colour order. Unsupported encodings and truncated buffers raise ``ValueError``
    so callers can count a malformed frame instead of publishing a misleading one.
    """
    try:
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(getattr(msg, "encoding", "") or "").lower()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("image message has invalid metadata") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    channels = _SUPPORTED.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    row_bytes = width * channels
    try:
        step = int(getattr(msg, "step", 0) or row_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("image step is invalid") from exc
    if step < row_bytes:
        raise ValueError(f"image step {step} is smaller than row width {row_bytes}")
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("image data is not a byte buffer") from exc
    # ``step`` describes every row, including its padding. ROS image buffers are
    # therefore ``height * step`` bytes long; requiring the complete final row
    # prevents a truncated padded frame from being reshaped as if it were valid.
    required = height * step
    if raw.size < required:
        raise ValueError(f"image buffer has {raw.size} bytes but needs at least {required}")
    pixels = raw[:required].reshape(height, step)[:, :row_bytes]
    return pixels.reshape(height, width, channels), encoding
