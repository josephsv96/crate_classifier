# from os.path import samefile

import numpy as np
import matplotlib.pyplot as plt

from utils import load_npy
from postprocessing import BandPass
from utils import get_custom_cmap

mod_2_net_out = load_npy("mod_2_ResNet18_46k_predictions.npy")
CMAP = get_custom_cmap()

index = 13
sample_annot = mod_2_net_out[index, :, :, :]

# Postprocessing
sample_filtered = BandPass.low_pass(sample_annot, 0.9)


# Results
sub_channel = 7
plt.subplot(2, 2, 1)
plt.imshow(np.argmax(sample_annot, axis=-1), cmap=CMAP)
plt.subplot(2, 2, 2)
plt.imshow(sample_annot[:, :, sub_channel])
plt.subplot(2, 2, 3)
plt.imshow(np.argmax(sample_filtered, axis=-1), cmap=CMAP)
plt.subplot(2, 2, 4)
plt.imshow(sample_filtered[:, :, sub_channel])

plt.show()
