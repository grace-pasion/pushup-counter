import os
import kagglehub

# -----------------------------
# 1. Output folder
# -----------------------------
output_dir = "pushups_only"
os.makedirs(output_dir, exist_ok=True)
print("Created output folder:", output_dir)

# -----------------------------
# 2. List of push-up files to download
# -----------------------------
# We'll just pick the first few for demonstration; you can expand the list
pushup_files = [
    "UCF101/PushUps/v_PushUps_g01_c01.avi",
    "UCF101/PushUps/v_PushUps_g01_c02.avi",
    "UCF101/PushUps/v_PushUps_g02_c01.avi",
    "UCF101/PushUps/v_PushUps_g02_c02.avi",
]

dataset_name = "matthewjansen/ucf101-action-recognition"

# -----------------------------
# 3. Download each push-up video
# -----------------------------
for file_path in pushup_files:
    print("Downloading:", file_path)
    
    local_path = kagglehub.download_file(
        dataset_name,
        file_path=file_path
    )
    
    # Move to output folder
    filename = os.path.basename(file_path)
    os.rename(local_path, os.path.join(output_dir, filename))

print("DONE! Push-up videos saved to:", output_dir)
