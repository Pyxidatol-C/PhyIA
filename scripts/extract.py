import numpy as np
from tqdm import trange
from typing import List, Tuple


def main_axis(im: np.ndarray, pbar: bool = False) -> Tuple[int, int]:
    """Find the main axis of the diffraction pattern.

    :param im: The image as an np.ndarray (e.g., returned by cv.imread).
    :param pbar: (Optional, defaults to False) whether a progress bar should be displayed.
    :return: The main axis of the diffraction pattern, i.e., the line on which the sum of the pixels' intensity is max.
    """
    # Separate the red layer from BGR
    im_r = im[:, :, 2]
    y_dim, x_dim = im_r.shape  # y: vertical; x: horizontal

    i_sum_max = -1  # Current maximum sum of intensity
    (a_best, b_best) = (None, None)

    # The two extremities
    for y_left in (trange if pbar else range)(y_dim):
        for y_right in range(y_dim):
            # Equation: y = a * x + b
            a = (y_right - y_left) / x_dim  # slope
            b = y_left

            i_sum = 0  # Current sum of intensity

            for x in range(x_dim):
                y = int(round(a * x + b))
                i_sum += im_r[y, x]

            if i_sum_max < i_sum:
                i_sum_max = i_sum
                (a_best, b_best) = (a, b)

    if i_sum_max == -1:
        raise ValueError("No main axis found")

    return a_best, b_best


def pattern(im: np.ndarray, axis: Tuple[int, int]) -> List[int]:
    """Get the diffraction pattern as an 1D array.

    :param im: The image containing the diffraction pattern.
    :param axis: The equation (m, k where y = m * x + k) of the main axis of the diffraction pattern.
    :return: 1D array containing the intensity along the main axis of the diffraction pattern.
    """
    diffraction = []
    for x in range(im.shape[1]):
        y = int(round(axis[0] * x + axis[1]))
        diffraction.append(im[y, x, 2])
    return diffraction

