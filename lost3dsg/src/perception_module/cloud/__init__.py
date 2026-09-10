"""Cloud Perception Backend for Open-Vocabulary 3D Scene Graph Generation.

Supports:
- Modal serverless GPU backend (L4/T4) for zero-local-model inference.
- Managed API backends (Fal.ai, Replicate).
- Local fallback backend for offline evaluation.
"""

from .client import (
    LocalPerceptionBackend,
    ManagedPerceptionBackend,
    ModalPerceptionBackend,
    PerceptionBackend,
    get_perception_backend,
)

__all__ = [
    "PerceptionBackend",
    "ModalPerceptionBackend",
    "ManagedPerceptionBackend",
    "LocalPerceptionBackend",
    "get_perception_backend",
]
