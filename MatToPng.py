# 1. Imports & Paths
import os, glob
import numpy as np
from PIL import Image
import scipy.io
import h5py

INPUT_DIR  = "E://PROJECTS and RP's//Tumor Detection 6th sem//Dataset//brainTumorDataPublic_2299-3064"
OUTPUT_DIR = "E://PROJECTS and RP's//Tumor Detection 6th sem//Dataset//png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Find all .mat files
mat_files = glob.glob(os.path.join(INPUT_DIR, '*.mat'))
print(f"🔍 Found {len(mat_files)} .mat files in {INPUT_DIR}\n")

# 3. Conversion loop
for mat_path in mat_files:
    fname = os.path.basename(mat_path)
    print(f"🛠  Processing {fname}...")

    try:
        # Try loading with scipy
        mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if 'cjdata' in mat:
            cj = mat['cjdata']
            img = cj.image if hasattr(cj, 'image') else cj['image']
        else:
            raise KeyError("No 'cjdata' in scipy loadmat result")
    except Exception as e1:
        try:
            # Try loading with h5py
            with h5py.File(mat_path, 'r') as f:
                if 'cjdata' not in f:
                    raise KeyError(f"'cjdata' not in {list(f.keys())}")
                # Try reading image inside cjdata
                cjdata = f['cjdata']
                # HDF5 groups are accessed using keys as bytes or strings
                if isinstance(cjdata, h5py.Group) and 'image' in cjdata:
                    img = np.array(cjdata['image'])
                else:
                    raise TypeError("Cannot read image from cjdata")
        except Exception as e2:
            print(f"❌ Failed to load {fname}:\n   scipy error: {e1}\n   h5py error: {e2}\n")
            continue

    # Normalize image to 0-255
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min())
    img_uint8 = (img * 255).astype(np.uint8)

    # Save as PNG
    out_name = os.path.splitext(fname)[0] + '.png'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    Image.fromarray(img_uint8).save(out_path)
    print(f"✅ Saved: {out_path}\n")

print("🎉 All done! Check your PNGs in", OUTPUT_DIR)