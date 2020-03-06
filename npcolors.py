# -*- coding: utf-8 -*-
import numpy as np

''' functions copied from matplotlib.colors for debugging '''


def rgb_to_hsv(arr):
    """
    Convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_shape = arr.shape
    arr = np.array(
        arr, copy=False,
        dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )
    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max
    out = out.reshape(in_shape)
    return out



def hsv_to_rgb(hsv):
    """
    Convert hsv values to rgb.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    in_shape = hsv.shape
    hsv = np.array(
        hsv, copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)



