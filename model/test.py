import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

data_path = Path("C:\\Users\\Tommy\\Documents\\GitHub\\ScapeGAN\\dataset\\processed\\24335img_BW.npz")

x = np.load(data_path)

# img = cv2.imread(x['images'][0])

img = x['images'][1]

img = (img + 1) * 127.5
img = img.astype('int')

print(img.shape)
plt.imshow(img)
plt.show()