import os
import random
import time
import traceback

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# CONFIG
# ------------------------------
UCF_CSV = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\good_quality_ucf_counts.csv"
UCF_VIDEO_DIRS = [
    r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\ucf101_dataset\medium_quality",
    r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\UCF101_push_ups\ucf101_dataset\good_quality"
]

KINETICS_CSV = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\Kinetics_push_ups\kinetic_counts.csv"
KINETICS_VIDEO_DIR = r"C:\Users\grpas\Downloads\push_up_project\pushup-counter\Kinetics_push_ups\kinetics_pushups"

OUT_MODEL = "cnn_pushup_model.pt"
OUT_PRED_CSV = "cnn_predictions.csv"

NUM_FRAMES = 32
IMG_SIZE = 112

BATCH_SIZE = 4
NUM_EPOCHS = 12
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ------------------------------
# CSV Loader
# ------------------------------
def load_csv_safe(path, dataset_name):
    df = pd.read_csv(path, header=0)
    df.columns = [c.strip() for c in df.columns]

    vid_col = next((c for c in df.columns if "video" in c.lower()), None)
    count_col = next((c for c in df.columns if "count" in c.lower()), None)

    df = df.rename(columns={vid_col: "video_id", count_col: "count"})
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["dataset"] = dataset_name
    return df


def load_all_data():
    df_ucf = load_csv_safe(UCF_CSV, "ucf")
    df_kin = load_csv_safe(KINETICS_CSV, "kinetics")
    return pd.concat([df_ucf, df_kin], ignore_index=True)


# ------------------------------
# Find video path
# ------------------------------
def find_video_path(video_id, dataset):
    search_dirs = UCF_VIDEO_DIRS if dataset == "ucf" else [KINETICS_VIDEO_DIR]

    # direct match
    for d in search_dirs:
        for ext in ["", ".mp4", ".avi", ".mov", ".mkv"]:
            p = os.path.join(d, video_id if ext == "" else (os.path.splitext(video_id)[0] + ext))
            if os.path.exists(p):
                return p

    # deep search
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[0] == os.path.splitext(video_id)[0]:
                    return os.path.join(root, f)

    return None


# ------------------------------
# Dataset
# ------------------------------
class VideoCountDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Uniform temporal sampling
        if total <= 0:
            indices = [0] * NUM_FRAMES
        else:
            indices = np.linspace(0, max(0, total - 1), NUM_FRAMES).astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
                continue

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame[:, :, ::-1]  # BGR → RGB
            frames.append(frame)

        cap.release()

        # frames: (T, H, W, 3)
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0

        # Convert to (C, T, H, W)
        arr = arr.transpose(3, 0, 1, 2)
        return arr


    def __getitem__(self, idx):
        row = self.df.loc[idx]
        vid = row["video_id"]
        dataset = row["dataset"]
        true_count = float(row["count"])

        path = find_video_path(vid, dataset)
        if path is None:
            frames = np.zeros((3, NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            frames = self._sample_frames(path)

        return (
            torch.from_numpy(frames).float(),
            torch.tensor([true_count], dtype=torch.float32),
            vid,
            path,
        )


# ------------------------------
# Collate FN (FIXED)
# ------------------------------
def collate_fn(batch):
    frames = torch.stack([item[0] for item in batch])
    counts = torch.stack([item[1] for item in batch])
    vids = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    return frames, counts, vids, paths


# ------------------------------
# Tiny 3D CNN
# ------------------------------
class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


# ------------------------------
# Training
# ------------------------------
def train_epoch(model, loader, optim, loss_fn, epoch):
    model.train()
    total_loss = 0
    total_mae = 0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Training", ncols=100)

    for frames, counts, vids, paths in pbar:
        frames = frames.to(DEVICE)
        counts = counts.to(DEVICE)

        optim.zero_grad()
        preds = model(frames)
        loss = loss_fn(preds, counts)
        loss.backward()
        optim.step()

        total_loss += loss.item() * frames.size(0)
        total_mae += torch.sum(torch.abs(preds - counts)).item()
        n += frames.size(0)

        pbar.set_postfix({"loss": total_loss/n, "mae": total_mae/n})

    return total_loss/n, total_mae/n


# ------------------------------
# Evaluation
# ------------------------------
def eval_model(model, loader, save_csv=None):
    model.eval()
    results = []

    pbar = tqdm(loader, desc="Evaluating", ncols=100)

    with torch.no_grad():
        for frames, counts, vids, paths in pbar:
            frames = frames.to(DEVICE)
            preds = model(frames).cpu().numpy().flatten()
            counts_np = counts.numpy().flatten()

            for vid, path, p, t in zip(vids, paths, preds, counts_np):
                results.append({
                    "video_id": vid,
                    "found_path": path,
                    "true_count": float(t),
                    "pred_count": float(p)
                })

    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(save_csv, index=False)
    return df


# ------------------------------
# MAIN
# ------------------------------
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    df = load_all_data()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["dataset"]
    )

    train_loader = DataLoader(
        VideoCountDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        VideoCountDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Tiny3DCNN().to(DEVICE)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_mae = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae = train_epoch(model, train_loader, optim, loss_fn, epoch)

        val_df = eval_model(model, test_loader)
        val_mae = float(np.mean(np.abs(val_df["pred_count"] - val_df["true_count"])))

        print(f"Epoch {epoch}: train_loss={train_loss:.3f}, train_mae={train_mae:.3f}, val_mae={val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"Saved best model (val MAE={val_mae:.3f}) → {OUT_MODEL}")

    final_df = eval_model(model, test_loader, save_csv=OUT_PRED_CSV)
    print("Final predictions saved:", OUT_PRED_CSV)


if __name__ == "__main__":
    main()
