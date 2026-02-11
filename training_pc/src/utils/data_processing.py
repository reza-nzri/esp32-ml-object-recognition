import numpy as np
import pandas as pd
import os
import glob
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


class DataProcessor:
    def __init__(self, max_dist=15.0, steps=163):
        self.max_dist = max_dist
        self.steps = steps

    def clean_scan(self, raw_distances):
        return median_filter(raw_distances, size=3)

    def process_file(self, file_path, label_idx):
        df = pd.read_csv(file_path, comment="#").dropna(subset=["distance_cm"])

        df.loc[df["distance_cm"] > self.max_dist, "distance_cm"] = np.nan

        df["distance_cm"] = df.groupby("scan_index")["distance_cm"].transform(
            lambda x: x.interpolate().fillna(self.max_dist)
        )

        scans, labels = [], []

        for s_id in df["scan_index"].unique():
            scan = df[df["scan_index"] == s_id]["distance_cm"].values
            cleaned = self.clean_scan(scan)
            padded = np.pad(cleaned, (0, max(0, self.steps - len(cleaned))), mode='edge')[:self.steps]

            scans.append(padded / self.max_dist)
            labels.append(label_idx)

        return np.array(scans), np.array(labels)

    def load_from_folders(self, base_path, shape_dirs):
        X_all, y_all = [], []
        sorted_shapes = sorted(shape_dirs.keys())

        for label_idx, shape_name in enumerate(sorted_shapes):
            folder_path = os.path.join(base_path, shape_dirs[shape_name])
            file_pattern = os.path.join(folder_path, "*.csv")
            files = glob.glob(file_pattern)

            if not files:
                print(f"Warning: No files found in {folder_path}")
                continue

            print(f"Loading {len(files)} file(s) for '{shape_name}' (ID: {label_idx})...")

            for file_path in files:
                scans, labels = self.process_file(file_path, label_idx)
                X_all.append(scans)
                y_all.append(labels)

        return np.concatenate(X_all)[..., np.newaxis], np.concatenate(y_all)


class SmartAugmentor:
    @staticmethod
    def rotate_scan(scan, shift_range=20):
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(scan, shift, axis=0)

    @staticmethod
    def add_noise(scan, noise_level=0.02):
        return scan + np.random.normal(0, noise_level, scan.shape)

    @staticmethod
    def jitter_scale(scan, factor=0.05):
        return scan * (1 + np.random.uniform(-factor, factor))