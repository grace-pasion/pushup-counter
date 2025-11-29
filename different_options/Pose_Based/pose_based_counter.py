# pose_based_counter.py
"""
Pose-based push-up counter (robust + CPU-friendly).
Saves per-video predictions to predictions.csv.

Config: edit the path constants below if needed.
Run:
    python pose_based_counter.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import mediapipe as mp
from sklearn.model_selection import train_test_split
import time
import traceback

# ------------------------------
# CONFIG — CHANGE THESE PATHS
# ------------------------------
UCF_CSV = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\good_quality_ucf_counts.csv"
UCF_VIDEO_DIRS = [
    r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\ucf101_dataset\medium_quality",
    r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\ucf101_dataset\good_quality"
]

KINETICS_CSV = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\Kinetics_push_ups\kinetic_counts.csv"
KINETICS_VIDEO_DIR = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\Kinetics_push_ups\kinetics_pushups"

# Sampling settings (increase FRAME_SKIP to run faster)
FRAME_SKIP = 2          # read every Nth frame (1 = every frame, 2 = every other frame)
MAX_FRAMES = 800        # safety cap (stop after reading this many frames)
ANGLE_SIDE = "left"     # 'left' or 'right' - which arm to use for angle
VALLEY_PROMINENCE = 0.15
VALLEY_DISTANCE = 8     # min frames between valleys (after sampling)

OUTPUT_PRED_CSV = "predictions.csv"

# ------------------------------
# UTIL: READ + NORMALIZE CSV
# ------------------------------
def load_csv_safe(path, dataset_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, header=0)

    # Trim whitespace in column names and values
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Acceptable column keys for video id and count
    possible_vid = ["video_id", "video", "filename", "file", "video_name", "videoId"]
    possible_count = ["count", "counts", "label_count"]

    # Find and rename (video id)
    found_vid = None
    for p in possible_vid:
        if p in df.columns:
            found_vid = p
            break
    if found_vid is None:
        raise ValueError(f"{dataset_name} CSV missing a video-id column. Columns: {list(df.columns)}")

    df = df.rename(columns={found_vid: "video_id"})

    # Find and rename (count)
    found_count = None
    for p in possible_count:
        if p in df.columns:
            found_count = p
            break
    if found_count is None:
        # maybe there is a column named 'count' but with trailing spaces etc (already trimmed),
        # if still missing then fail
        raise ValueError(f"{dataset_name} CSV missing a count column. Columns: {list(df.columns)}")

    df = df.rename(columns={found_count: "count"})

    # Ensure label exists (not required but keep)
    if "label" not in df.columns:
        df["label"] = dataset_name

    # Clean values: strip strings
    df["video_id"] = df["video_id"].astype(str).str.strip()
    # Make sure count is numeric
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Add dataset column
    df["dataset"] = dataset_name

    return df

# ------------------------------
# LOAD CSVs + MERGE
# ------------------------------
def load_all_data():
    df_ucf = load_csv_safe(UCF_CSV, "ucf")
    df_kin = load_csv_safe(KINETICS_CSV, "kinetics")
    df = pd.concat([df_ucf, df_kin], ignore_index=True)
    # Drop duplicates if any (keep first)
    df = df.drop_duplicates(subset=["video_id", "dataset", "count"]).reset_index(drop=True)
    return df

# ------------------------------
# FIND VIDEO PATH (robust)
# ------------------------------
def find_video_path(video_id, dataset):
    # If video_id is already an absolute path, check it directly
    if os.path.isabs(video_id) and os.path.exists(video_id):
        return video_id

    # try relative to KINETICS_VIDEO_DIR and UCF dirs
    if dataset == "ucf":
        for d in UCF_VIDEO_DIRS:
            cand = os.path.join(d, video_id)
            if os.path.exists(cand):
                return cand
            # also try .mp4/.avi variations if extension missing
            base, ext = os.path.splitext(video_id)
            for ext_try in [".avi", ".mp4", ".MOV", ".mkv"]:
                cand2 = os.path.join(d, base + ext_try)
                if os.path.exists(cand2):
                    return cand2
    elif dataset == "kinetics":
        cand = os.path.join(KINETICS_VIDEO_DIR, video_id)
        if os.path.exists(cand):
            return cand
        base, ext = os.path.splitext(video_id)
        for ext_try in [".avi", ".mp4", ".MOV", ".mkv"]:
            cand2 = os.path.join(KINETICS_VIDEO_DIR, base + ext_try)
            if os.path.exists(cand2):
                return cand2

    # if not found, attempt to search recursively (expensive) but limited depth to avoid long stalls
    search_dirs = UCF_VIDEO_DIRS + [KINETICS_VIDEO_DIR]
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # try direct file name match in this directory tree
        for root, _, files in os.walk(d):
            if video_id in files:
                return os.path.join(root, video_id)
            # also try base name matches
            base_name = os.path.splitext(video_id)[0]
            for f in files:
                if os.path.splitext(f)[0] == base_name:
                    return os.path.join(root, f)
    return None

# ------------------------------
# POSE EXTRACTION (MediaPipe)
# ------------------------------
mp_pose = mp.solutions.pose

def compute_elbow_angles(video_path, frame_skip=FRAME_SKIP, max_frames=MAX_FRAMES, side=ANGLE_SIDE):
    """
    Returns numpy array of elbow angles per sampled frame.
    side: 'left' or 'right' (left uses landmarks 11,13,15; right uses 12,14,16)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    # select indices for left/right landmarks
    if side.lower().startswith("r"):
        S_IDX, E_IDX, W_IDX = 12, 14, 16
    else:
        S_IDX, E_IDX, W_IDX = 11, 13, 15

    angles = []
    frame_idx = 0
    processed = 0

    # Initialize Pose once and reuse
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # sample every Nth frame
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # limit processed frames
            if processed >= max_frames:
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(frame_rgb)
            except Exception:
                # If MediaPipe errors (rare), append 0 and continue
                traceback.print_exc()
                angles.append(0.0)
                frame_idx += 1
                processed += 1
                continue

            if not res.pose_landmarks:
                angles.append(0.0)
            else:
                try:
                    l = res.pose_landmarks.landmark
                    s = np.array([l[S_IDX].x, l[S_IDX].y])
                    e = np.array([l[E_IDX].x, l[E_IDX].y])
                    w = np.array([l[W_IDX].x, l[W_IDX].y])

                    v1 = s - e
                    v2 = w - e
                    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
                    angle = float(np.degrees(np.arccos(cosang)))
                    angles.append(angle)
                except Exception:
                    # If any landmark missing / index error, append 0
                    angles.append(0.0)

            frame_idx += 1
            processed += 1

    cap.release()
    return np.array(angles, dtype=float)

# ------------------------------
# COUNT PUSH-UPS FROM ANGLE SIGNAL
# ------------------------------
def count_pushups_from_angles(angles, prominence=VALLEY_PROMINENCE, distance=VALLEY_DISTANCE):
    """
    angles: numpy array of elbow angles (degrees) sampled over time
    returns: integer number of detected push-ups (valleys)
    """
    if angles is None or len(angles) == 0:
        return 0

    # If constant or near-constant signal, return 0
    if np.nanstd(angles) < 1e-3:
        return 0

    # Smooth a little with rolling median to reduce noise (simple)
    try:
        window = max(3, int(len(angles) * 0.02))  # 2% of length or min 3
        if window % 2 == 0:
            window += 1
        # median smoothing
        pad = window // 2
        padded = np.pad(angles, (pad, pad), mode="edge")
        sm = np.array([np.median(padded[i:i+window]) for i in range(len(angles))])
    except Exception:
        sm = angles

    a = sm
    # Normalize to [0,1]
    a_min, a_max = a.min(), a.max()
    if (a_max - a_min) < 1e-6:
        a_norm = np.zeros_like(a)
    else:
        a_norm = (a - a_min) / (a_max - a_min)

    # find valleys (push-up bottoms) by finding peaks in negative signal
    try:
        valleys, props = find_peaks(-a_norm, prominence=prominence, distance=distance)
        count = len(valleys)
    except Exception:
        # fallback more permissive
        valleys, props = find_peaks(-a_norm, prominence=0.1, distance=max(4, distance//2))
        count = len(valleys)

    return int(count)

# ------------------------------
# PROCESS SINGLE VIDEO (wrapper)
# ------------------------------
def process_video(video_path):
    try:
        angles = compute_elbow_angles(video_path)
        pred = count_pushups_from_angles(angles)
    except Exception:
        traceback.print_exc()
        pred = 0
    return pred

# ------------------------------
# EVALUATION + SAVE
# ------------------------------
def evaluate_and_save(df_test, out_csv=OUTPUT_PRED_CSV, save_angles_dir=None):
    results = []
    total = len(df_test)
    start_time = time.time()

    for idx, row in df_test.iterrows():
        vid = row.get("video_id")
        true_count = int(row.get("count", 0))
        dataset = row.get("dataset", "unknown")

        if not isinstance(vid, str) or vid.strip() == "":
            print(f"[SKIP] row {idx} missing video_id. Row: {row.to_dict()}")
            continue
        vid = vid.strip()

        path = find_video_path(vid, dataset)
        if path is None:
            print(f"[MISSING] Video not found: {vid} (dataset={dataset})")
            results.append({"video_id": vid, "dataset": dataset, "true_count": true_count, "pred_count": None, "found_path": None})
            continue

        t0 = time.time()
        pred = process_video(path)
        t1 = time.time()

        print(f"[{idx+1}/{total}] {vid}  dataset={dataset}  pred={pred}  true={true_count}  time={t1-t0:.2f}s  path={path}")
        results.append({"video_id": vid, "dataset": dataset, "true_count": true_count, "pred_count": pred, "found_path": path})

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_csv, index=False)
    elapsed = time.time() - start_time

    # compute metrics ignoring missing preds
    valid = df_res[df_res["pred_count"].notna()]
    if len(valid) > 0:
        preds = valid["pred_count"].astype(float).values
        trues = valid["true_count"].astype(float).values
        mae = float(np.mean(np.abs(preds - trues)))
        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    else:
        mae = rmse = None

    print("\n==== EVAL SUMMARY ====")
    print(f"Videos evaluated   : {len(valid)} / {len(df_test)}")
    print(f"Elapsed time (s)   : {elapsed:.1f}")
    if mae is not None:
        print(f"MAE                : {mae:.3f}")
        print(f"RMSE               : {rmse:.3f}")
    else:
        print("No valid predictions to compute MAE/RMSE.")
    print(f"Predictions saved to: {os.path.abspath(out_csv)}")

    return df_res

# ------------------------------
# MAIN
# ------------------------------
def main():
    print("Loading CSVs...")
    df = load_all_data()
    print(f"Total videos found in CSVs: {len(df)}")

    # sanity: show columns
    print("Columns:", df.columns.tolist())

    # if video_id not present, raise (shouldn't happen due to load_csv_safe)
    if "video_id" not in df.columns or "count" not in df.columns:
        raise RuntimeError("Required columns missing after load. Columns: " + ", ".join(df.columns))

    # Split 80/20 preserving dataset proportion
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["dataset"] if "dataset" in df.columns else None)

    print(f"Train size = {len(train_df)}  Test size = {len(test_df)}")
    print("Starting pose-based evaluation on test set...\n")

    res_df = evaluate_and_save(test_df)

    print("\nDone.")

if __name__ == "__main__":
    main()
