import imageio.v2 as imageio
import numpy as np

def write_png(path, rgba_uint8):
    if rgba_uint8 is None:
        return False

    arr = np.asarray(rgba_uint8)
    if arr.ndim < 2:
        return False

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    imageio.imwrite(path, arr)
    return True