import cv2
import numpy as np
import matplotlib.pyplot as plt
from extract import main_axis, pattern
from multiprocessing.dummy import Pool


def save_pattern(f: str):
    im: np.ndarray = cv2.imread(f"raw-data/{f}.png")
    m, k = main_axis(im)

    diffraction = pattern(im, (m, k))

    fig = plt.imshow(np.array([diffraction] * 500))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(f"processed-data/{f}.png",
                bboxinches='tight',
                pad_inches=0,
                dpi=512)


if __name__ == '__main__':
    pool = Pool(12)
    imgs = [f"slits/{i}" for i in (40, 50, 100, 120, 280, 400)] + \
           [f"wires/{j}" for j in (38, 50, 76, 100, 120, 150)]
    pool.map(save_pattern, imgs)
