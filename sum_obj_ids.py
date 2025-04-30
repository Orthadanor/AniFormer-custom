import os
import re
import numpy as np

##############################################
# A script to scan your datasets/DFAUST_sequence_data_all/ directory, 
# extract all valid (subject_idx, sequence_idx, frame_idx) triplets from filenames, 
# and save them as a .npy file for use in your SMPL_sequence loader.
##############################################


# Path to your .obj files
obj_dir = './datasets/DFAUST_sequence_data_all'

# Prepare regex pattern and result list
triplets = []
pattern = re.compile(r'(\d+)_(\d+)_(\d+)\.obj')

# Loop through .obj files
for fname in os.listdir(obj_dir):
    match = pattern.match(fname)
    if match:
        triplet = tuple(map(int, match.groups()))  # (subject_idx, sequence_idx, frame_idx)
        triplets.append(triplet)

triplets = sorted(triplets)  # Optional: for consistency

# Save to .npy file
save_path = os.path.join(obj_dir, 'valid_triplets.npy')
np.save(save_path, triplets)

print(f"✅ Saved {len(triplets)} valid triplets to {save_path}")
