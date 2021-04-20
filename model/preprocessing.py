import numpy as np
import pandas as pd
from pathlib import Path

import sys
import cv2

DATA_SEGMENTS = ['train', 'test', 'pred']
TO_GRAYSCALE = True
IMG_DIMS = (128,128,1 + 2*(1-TO_GRAYSCALE))
SAVE_DIR = Path('../dataset/processed') if len(sys.argv) < 2 else sys.argv[1]


def load_dataset_folder(path, target_size=IMG_DIMS):

    img_paths = path.glob('*.jpg')

    proc_function = cv2.COLOR_BGR2RGB if not TO_GRAYSCALE else cv2.IMREAD_GRAYSCALE
    images = [cv2.imread(str(path), proc_function) for path in img_paths]
    images = [cv2.resize(img, target_size[0:2], interpolation=cv2.INTER_AREA) for img in images]
    images = np.array(images).astype('float32')
    images = (images - 127.5) / 127.5

    if not len(images.shape) == 4:
        images = np.expand_dims(images, -1)

    return images


if __name__ == "__main__":

    SAVE_DIR.mkdir(exist_ok=True)

    images = []
    labels = []

    for segment in DATA_SEGMENTS:

        image_sets = [load_dataset_folder(subdir) for subdir in Path(f'../dataset/seg_{segment}/seg_{segment}').glob('*')]

        images += image_sets
        labels += [i for i in range(6) for _ in image_sets[i]] if not segment == 'pred' \
                  else [6] * len(image_sets[0])

    labels = np.array(labels)
    images = np.concatenate(images, axis=0)

    save_path = SAVE_DIR / f'{images.shape[0]}img_{"C" if not TO_GRAYSCALE else "BW"}.npz'
    np.savez(save_path, images=images, labels=labels)


    log_message = (f'Prepared {images.shape[0]} images in {"grayscale" if TO_GRAYSCALE else "color"} from '
                   f'segments [{", ".join(DATA_SEGMENTS)}]. \n'
                   f'{save_path.stat().st_size / (10**6):.2f}MB of data saved to {save_path}.')

    print(log_message)
