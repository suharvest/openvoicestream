import numpy as np


def test_pad_mel_axis_extends_short_matcha_tensors():
    from voxedge.backends.jetson import matcha_trt

    arr = np.ones((1, matcha_trt.MEL_DIM, 64), dtype=np.float32)

    padded = matcha_trt._pad_mel_axis(arr)

    assert padded.shape == (1, matcha_trt.MEL_DIM, matcha_trt.MIN_MEL_FRAMES)
    np.testing.assert_array_equal(padded[:, :, :64], arr)
    assert np.all(padded[:, :, 64:] == 0)


def test_pad_mel_axis_leaves_supported_shapes_unchanged():
    from voxedge.backends.jetson import matcha_trt

    arr = np.ones((1, matcha_trt.MEL_DIM, matcha_trt.MIN_MEL_FRAMES), dtype=np.float32)

    assert matcha_trt._pad_mel_axis(arr) is arr
