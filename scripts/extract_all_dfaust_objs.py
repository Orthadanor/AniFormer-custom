import os
import h5py

# Load subject -> [sequences] mapping
subject_seq_map = {
    "50002": ["chicken_wings", "hips", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50004": ["chicken_wings", "hips", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot_bugfix", "shake_arms", "shake_hips", "shake_shoulders"],
    "50007": ["chicken_wings", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50009": ["chicken_wings", "hips", "jiggle_on_toes", "jumping_jacks",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_hips"],
    "50020": ["chicken_wings", "hips", "jiggle_on_toes", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "personal_move", "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50021": ["chicken_wings", "hips", "knees", "light_hopping_stiff", "one_leg_jump",
              "one_leg_loose", "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50022": ["hips", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50025": ["chicken_wings", "hips", "jiggle_on_toes", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50026": ["chicken_wings", "hips", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"],
    "50027": ["hips", "jiggle_on_toes", "jumping_jacks", "knees",
              "light_hopping_loose", "light_hopping_stiff", "one_leg_jump", "one_leg_loose",
              "punching", "running_on_spot", "shake_arms", "shake_hips", "shake_shoulders"]
}

# Setup
input_paths = ["registrations_m.hdf5", "registrations_f.hdf5"]
output_dir = "./datasets/DFAUST_sequence_data_all"
os.makedirs(output_dir, exist_ok=True)

# Subject and sequence index mappings
subject_idx_map = {sid: i for i, sid in enumerate(subject_seq_map)}
all_sequences = sorted({seq for seqs in subject_seq_map.values() for seq in seqs})
sequence_idx_map = {seq: i for i, seq in enumerate(all_sequences)}

# Write function
def write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces + 1:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')

# Process each file
for file_path in input_paths:
    with h5py.File(file_path, 'r') as f:
        faces = f["faces"][()]
        for sid, sequences in subject_seq_map.items():
            for seq in sequences:
                key = f"{sid}_{seq}"
                if key not in f:
                    print(f"[WARNING] Missing: {key}")
                    continue
                verts_seq = f[key][()].transpose(2, 0, 1)  # shape (frames, 6890, 3)
                subject_idx = subject_idx_map[sid]
                sequence_idx = sequence_idx_map[seq]
                for frame_idx, verts in enumerate(verts_seq):
                    filename = f"{subject_idx}_{sequence_idx}_{frame_idx}.obj"
                    full_path = os.path.join(output_dir, filename)
                    write_obj(full_path, verts, faces)
                print(f"✓ {key}: {len(verts_seq)} frames")

print("✅ Extraction complete.")
