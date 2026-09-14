"""Evaluate RoMa + USAC-MAGSAC on sampled training pairs."""

import csv
from glob import glob
import os
import random

import cv2
import numpy as np
import torch
from romatch import roma_outdoor

from common import (
    compute_maa,
    estimate_pose_from_matches,
    load_calibration,
    pose_error,
    quaternion_from_matrix,
    read_covisibility_data,
    relative_ground_truth,
)

RANDOM_SEED = 37
MAX_PAIRS_PER_SCENE = 10
DATA_ROOT = "Data/train"


def load_scaling_factors(path):
    scales = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            scales[row[1]] = float(row[2])
    return scales


def run_roma(model, image_a_path, image_b_path, image_a, image_b, device):
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    with torch.no_grad():
        warp, certainty = model.match(image_a_path, image_b_path, device=device)
        matches, certainty = model.sample(warp, certainty)
        kpts_a, kpts_b = model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
    return kpts_a.cpu().numpy(), kpts_b.cpu().numpy()


def main():
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = roma_outdoor(device=device)

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
            path1 = f"{scene_path}/images/{id1}.jpg"
            path2 = f"{scene_path}/images/{id2}.jpg"
            img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

            kpts1, kpts2 = run_roma(model, path1, path2, img1, img2, device)
            result = estimate_pose_from_matches(kpts1, kpts2, calibration[id1].K, calibration[id2].K, threshold=0.25)
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
