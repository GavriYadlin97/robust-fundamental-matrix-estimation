"""Evaluate LoFTR + USAC-MAGSAC on the training scenes."""

import csv
from glob import glob
import os
import random

import cv2
import numpy as np
import torch

from LoFTR.src.loftr import LoFTR, default_cfg
from common import (
    compute_maa,
    estimate_pose_from_matches,
    load_calibration,
    pad_to_divisor,
    pose_error,
    quaternion_from_matrix,
    read_covisibility_data,
    relative_ground_truth,
)

RANDOM_SEED = 37
MAX_PAIRS_PER_SCENE = 10
DATA_ROOT = "Data/train"
LOFTR_CHECKPOINT = "LoFTR/weights/outdoor_ds.ckpt"


def load_scaling_factors(path):
    scales = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            scales[row[1]] = float(row[2])
    return scales


def run_loftr(model, img0, img1, device):
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray0, _, _ = pad_to_divisor(gray0, 8)
    gray1, _, _ = pad_to_divisor(gray1, 8)

    batch = {
        "image0": torch.from_numpy(gray0)[None, None].to(device),
        "image1": torch.from_numpy(gray1)[None, None].to(device),
    }
    with torch.no_grad():
        model(batch)
    return batch["mkpts0_f"].cpu().numpy(), batch["mkpts1_f"].cpu().numpy()


def main():
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LoFTR(config=default_cfg)
    checkpoint = torch.load(LOFTR_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    scaling = load_scaling_factors(f"{DATA_ROOT}/scaling_factors.csv")
    scene_scores = {}

    for scene_path in sorted(p for p in glob(f"{DATA_ROOT}/*") if os.path.isdir(p)):
        scene = os.path.basename(scene_path)
        calibration = load_calibration(f"{scene_path}/calibration.csv")
        covisibility = read_covisibility_data(f"{scene_path}/pair_covisibility.csv")
        pairs = list(covisibility)
        random.shuffle(pairs)
        pairs = pairs[:MAX_PAIRS_PER_SCENE]

        rotation_errors, translation_errors = [], []
        print(f"\n=== {scene}: {len(pairs)} sampled pairs ===")

        for pair in pairs:
            id1, id2 = pair.split("-")
            img0 = cv2.cvtColor(cv2.imread(f"{scene_path}/images/{id1}.jpg"), cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(cv2.imread(f"{scene_path}/images/{id2}.jpg"), cv2.COLOR_BGR2RGB)

            kpts0, kpts1 = run_loftr(model, img0, img1, device)
            result = estimate_pose_from_matches(kpts0, kpts1, calibration[id1].K, calibration[id2].K, threshold=0.2)
            if result is None:
                print(f"{pair}: skipped (insufficient robust matches)")
                continue

            _, _, R, T, _, _, _ = result
            q_gt, T_gt = relative_ground_truth(calibration, id1, id2)
            q = quaternion_from_matrix(R)
            err_q, err_t = pose_error(q_gt, T_gt, q, T, scaling[scene])
            rotation_errors.append(err_q)
            translation_errors.append(err_t)
            print(f"{pair}: rotation={err_q:.2f}°, translation={err_t:.2f} m")

        if rotation_errors:
            score = compute_maa(rotation_errors, translation_errors)
            scene_scores[scene] = score
            print(f"{scene} mAA: {score:.5f}")
        else:
            scene_scores[scene] = 0.0
            print(f"{scene} mAA: 0.00000")

    print("\n--- Summary ---")
    for scene, score in scene_scores.items():
        print(f"{scene}: {score:.5f}")
    print(f"Overall mean scene mAA: {np.mean(list(scene_scores.values())):.5f}")


if __name__ == "__main__":
    main()
