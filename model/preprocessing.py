import numpy as np
import pandas as pd

import cv2

import keras


IMG_DIMS = (150,150,3)

def load_dataset_folder(path):

    