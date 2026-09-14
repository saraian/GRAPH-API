import tempfile
import unittest
from pathlib import Path

import numpy as np

from .depth_codec import read_depth, write_shuffled


class DepthCodecTests(unittest.TestCase):
    def test_all_float_bits_survive_including_signed_zero_and_nan_payloads(self):
        rng = np.random.default_rng(12)
        bits = rng.integers(0, 2**32, (32, 32), dtype=np.uint32)
        bits[0, :4] = [0, 0x80000000, 0x7f800000, 0x7fc00012]
        with tempfile.TemporaryDirectory() as tmp:
            write_shuffled(Path(tmp)/'frame.npz', bits.view(np.float32))
            actual = read_depth(tmp, 'frame', 'npz-shuffle-lossless')
            np.testing.assert_array_equal(bits, actual.view(np.uint32))

    def test_strided_input_and_old_formats(self):
        depth = np.arange(100, dtype=np.float32).reshape(10, 10)[:, ::2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_shuffled(root/'shuffle.npz', depth)
            np.savez_compressed(root/'zip.npz', depth=depth)
            np.save(root/'raw.npy', depth)
            for stem, codec in [('shuffle', 'npz-shuffle-lossless'), ('zip', 'npz-lossless'), ('raw', 'npy')]:
                np.testing.assert_array_equal(read_depth(root, stem, codec), depth)
            with self.assertRaises(ValueError):
                read_depth(root, 'raw', 'unsupported-codec')


if __name__ == '__main__':
    unittest.main()
