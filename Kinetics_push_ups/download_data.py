import os
import pandas as pd
import subprocess

csv_path = "train.csv"  # "train.csv", "val.csv", or "test.csv"
output_dir = "kinetics_pushups"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path, header=None,
                 names=["label", "youtube_id", "start_time", "end_time", "split", "dummy"])

pushups = df[df["label"].str.lower() == "push up"]

print(f"Found {len(pushups)} push-up clips.")

for idx, row in pushups.iterrows():
    youtube_id = row["youtube_id"]
    start = float(row["start_time"])
    end = float(row["end_time"])
    duration = end - start

    output_file = os.path.join(output_dir, f"{youtube_id}_{int(start)}_{int(end)}.mp4")

    if os.path.exists(output_file):
        print("Skipping (already exists):", output_file)
        continue

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={youtube_id}",
        "-f", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", output_file
    ]

    print("Downloading:", output_file)
    subprocess.run(cmd)
