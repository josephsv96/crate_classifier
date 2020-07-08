from utilities import load_npy
from pathlib import Path
import matplotlib.pyplot as plt

file = Path(
    "C:/Users/josep/Documents/work/crate_classifier/dataset/data_2/dataset/crate_1_annot/IMG_116.npy")

arr = load_npy(file)
print(arr.shape[0], arr.shape[1])

plt.imshow(arr)
plt.colorbar()
plt.show()
