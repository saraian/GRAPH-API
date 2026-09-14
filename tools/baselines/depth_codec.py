"""Lossless storage for native float32 metre depth; no depth quantization."""
from pathlib import Path

import numpy as np


def read_depth(root, stem, storage):
    root = Path(root)
    if storage not in ('npy', 'npz-lossless', 'npz-shuffle-lossless'):
        raise ValueError(f'Unknown depth storage: {storage}')
    if storage == 'npy':
        return np.load(root / (stem + '.npy'))
    with np.load(root / (stem + '.npz')) as data:
        if storage == 'npz-lossless':
            return data['depth']
        if storage == 'npz-shuffle-lossless':
            return np.ascontiguousarray(data['depth_bytes'].T).view('<f4').reshape(tuple(data['depth_shape']))
    raise ValueError(f'Unknown depth storage: {storage}')


def write_shuffled(path, depth):
    depth = np.ascontiguousarray(depth, dtype='<f4')
    shuffled = depth.view(np.uint8).reshape(-1, 4).T.copy()
    np.savez_compressed(path, depth_bytes=shuffled, depth_shape=np.asarray(depth.shape))
