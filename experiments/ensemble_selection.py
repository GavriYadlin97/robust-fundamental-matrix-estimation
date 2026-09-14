"""Exploratory validation-only ensemble of LoFTR and RoMa correspondences.

This reproduces the project idea of estimating three candidate geometries per image pair:
LoFTR only, RoMa only, and concatenated LoFTR+RoMa matches. Ground-truth pose is
used to select the best candidate, so this is an oracle-style validation experiment rather
than a deployable test-time selector.
"""

import csv
from glob import glob
import os
import random

import cv2
import numpy as np
import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from romatch import roma_outdoor

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
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

DATA_ROOT = "Data/train"
LOFTR_CHECKPOINT = "LoFTR/weights/outdoor_ds.ckpt"
MAX_PAIRS_PER_SCENE = 10
RANDOM_SEED = 37
ROTATION_RANGE = (1.0, 10.0)
TRANSLATION_RANGE = (0.2, 5.0)


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


def run_roma(model, path0, path1, img0, img1, device):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    with torch.no_grad():
        warp, certainty = model.match(path0, path1, device=device)
        matches, certainty = model.sample(warp, certainty)
        k0, k1 = model.to_pixel_coordinates(matches, h0, w0, h1, w1)
    return k0.cpu().numpy(), k1.cpu().numpy()


def normalized_error(rotation_error, translation_error):
    r = np.clip((rotation_error - ROTATION_RANGE[0]) / (ROTATION_RANGE[1] - ROTATION_RANGE[0]), 0.0, 1.0)
    t = np.clip((translation_error - TRANSLATION_RANGE[0]) / (TRANSLATION_RANGE[1] - TRANSLATION_RANGE[0]), 0.0, 1.0)
    return r + t


def evaluate_candidate(k0, k1, calibration, id0, id1, q_gt, t_gt, scale):
    result = estimate_pose_from_matches(k0, k1, calibration[id0].K, calibration[id1].K, threshold=0.2)
    if result is None:
        return None
    _, _, R, T, _, _, _ = result
    q = quaternion_from_matrix(R)
    err_q, err_t = pose_error(q_gt, t_gt, q, T, scale)
    return err_q, err_t


def main():
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loftr = LoFTR(config=default_cfg)
    checkpoint = torch.load(LOFTR_CHECKPOINT, map_location=device)
    loftr.load_state_dict(checkpoint["state_dict"])
    loftr = loftr.to(device).eval()
    roma = roma_outdoor(device=device)

    scaling = load_scaling_factors(f"{DATA_ROOT}/scaling_factors.csv")
    all_rot, all_trans = [], []

    for scene_path in sorted(p for p in glob(f"{DATA_ROOT}/*") if os.path.isdir(p)):
        scene = os.path.basename(scene_path)
        calibration = load_calibration(f"{scene_path}/calibration.csv")
        pairs = list(read_covisibility_data(f"{scene_path}/pair_covisibility.csv"))
        random.shuffle(pairs)
        pairs = pairs[:MAX_PAIRS_PER_SCENE]
        scene_rot, scene_trans = [], []

        print(f"\n=== {scene} ===")
        for pair in pairs:
            id0, id1 = pair.split("-")
            path0 = f"{scene_path}/images/{id0}.jpg"
            path1 = f"{scene_path}/images/{id1}.jpg"
            img0 = cv2.cvtColor(cv2.imread(path0), cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)

            l0, l1 = run_loftr(loftr, img0, img1, device)
            r0, r1 = run_roma(roma, path0, path1, img0, img1, device)
            q_gt, t_gt = relative_ground_truth(calibration, id0, id1)

            candidates = {
                "LoFTR": (l0, l1),
                "RoMa": (r0, r1),
                "Combined": (np.concatenate([l0, r0]), np.concatenate([l1, r1])),
            }

            scored = []
            for name, (k0, k1) in candidates.items():
                errors = evaluate_candidate(k0, k1, calibration, id0, id1, q_gt, t_gt, scaling[scene])
                if errors is not None:
                    err_q, err_t = errors
                    scored.append((normalized_error(err_q, err_t), name, err_q, err_t))
                    print(f"{pair} {name}: rot={err_q:.2f}°, trans={err_t:.2f} m")

            if not scored:
                continue
            _, best_name, best_rot, best_trans = min(scored)
            print(f"{pair} -> oracle selection: {best_name}")
            scene_rot.append(best_rot)
            scene_trans.append(best_trans)
            all_rot.append(best_rot)
            all_trans.append(best_trans)

        if scene_rot:
            print(f"{scene} oracle mAA: {compute_maa(scene_rot, scene_trans):.5f}")

    if all_rot:
        print(f"\nOverall oracle mAA: {compute_maa(all_rot, all_trans):.5f}")


if __name__ == "__main__":
    main()
