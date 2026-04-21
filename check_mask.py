import numpy as np
from PIL import Image
import glob
import os

mask_files = glob.glob('outputs/mask/*.png')
if mask_files:
    # Check the first mask file
    img = Image.open(mask_files[0])
    arr = np.array(img)
    print(f"File: {os.path.basename(mask_files[0])}")
    print(f"Unique values in Red channel (Semantic Class IDs): {np.unique(arr[:, :, 0])}")
    print(f"Unique values in Green channel (Instance IDs): {np.unique(arr[:, :, 1])}")
else:
    print("No mask files found.")
