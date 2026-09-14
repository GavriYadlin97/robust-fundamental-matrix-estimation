"""Generate a resumable Kaggle-style submission with RoMa + USAC-MAGSAC."""

import argparse
import csv
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from romatch import roma_outdoor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="Data/test.csv")
    parser.add_argument("--image-root", default="Data/test_images")
    parser.add_argument("--output", default="RoMa_submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = roma_outdoor(device=device)

    test_df = pd.read_csv(args.test_csv)
    processed_ids = set()
    if os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output)
            processed_ids = set(existing["sample_id"].astype(str))
        except Exception:
            processed_ids = set()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing image pairs"):
        sample_id = str(row["sample_id"])
        if sample_id in processed_ids:
            continue

        batch_id = str(row["batch_id"])
        image1_id = str(row["image_1_id"])
        image2_id = str(row["image_2_id"])
        image1_path = os.path.join(args.image_root, batch_id, f"{image1_id}.jpg")
        image2_path = os.path.join(args.image_root, batch_id, f"{image2_id}.jpg")

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        if image1 is None or image2 is None:
            continue

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        with torch.no_grad():
            warp, certainty = model.match(image1_path, image2_path, device=device)
            matches, certainty = model.sample(warp, certainty)
            kpts1, kpts2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)

        F = np.zeros((3, 3), dtype=np.float64)
        if kpts1.shape[0] >= 8:
            try:
                candidate, mask = cv2.findFundamentalMat(
                    kpts1.cpu().numpy(),
                    kpts2.cpu().numpy(),
                    cv2.USAC_MAGSAC,
                    0.2,
                    0.99999,
                    20000,
                )
                if candidate is not None and mask is not None and mask.astype(bool).sum() >= 8:
                    F = candidate
            except cv2.error:
                pass

        matrix_string = " ".join(map(str, F.flatten()))
        with open(args.output, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["sample_id", "fundamental_matrix"])
            writer.writerow([sample_id, matrix_string])
        processed_ids.add(sample_id)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
