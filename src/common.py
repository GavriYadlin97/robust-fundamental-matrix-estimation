"""Shared geometry and evaluation utilities for the matching experiments."""

from collections import namedtuple
import csv

import cv2
import numpy as np

Gt = namedtuple("Gt", ["K", "R", "T"])
EPS = 1e-15


def pad_to_divisor(img, divisor=8):
    h, w = img.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if img.ndim == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant")
    elif img.ndim == 3:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    else:
        padded = img
    return padded, pad_h, pad_w


def read_covisibility_data(filename):
    values = {}
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            values[row[1]] = float(row[2])
    return values


def load_calibration(filename):
    calibration = {}
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            camera_id = row[1]
            K = np.array([float(v) for v in row[2].split()]).reshape(3, 3)
            R = np.array([float(v) for v in row[3].split()]).reshape(3, 3)
            T = np.array([float(v) for v in row[4].split()])
            calibration[camera_id] = Gt(K=K, R=R, T=T)
    return calibration


def normalize_keypoints(keypoints, K):
    center = np.array([[K[0, 2], K[1, 2]]])
    focal = np.array([[K[0, 0], K[1, 1]]])
    return (keypoints - center) / focal


def compute_essential_matrix(F, K1, K2, kp1, kp2):
    E = (K2.T @ F @ K1).astype(np.float64)
    kp1n = normalize_keypoints(kp1, K1)
    kp2n = normalize_keypoints(kp2, K2)
    _, R, T, _ = cv2.recoverPose(E, kp1n, kp2n)
    return E, R, T


def quaternion_from_matrix(matrix):
    M = np.asarray(matrix, dtype=np.float64)[:3, :3]
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    K = np.array([
        [m00 - m11 - m22, 0.0, 0.0, 0.0],
        [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
        [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
        [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
    ]) / 3.0
    values, vectors = np.linalg.eigh(K)
    q = vectors[[3, 0, 1, 2], np.argmax(values)]
    return -q if q[0] < 0 else q


def pose_error(q_gt, T_gt, q, T, scale):
    q_gt = q_gt / (np.linalg.norm(q_gt) + EPS)
    q = q / (np.linalg.norm(q) + EPS)
    loss_q = max(EPS, 1.0 - np.sum(q * q_gt) ** 2)
    err_q = np.arccos(1.0 - 2.0 * loss_q) * 180.0 / np.pi

    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + EPS)
    err_t = min(
        np.linalg.norm(T_gt_scaled - T_scaled),
        np.linalg.norm(T_gt_scaled + T_scaled),
    )
    return err_q, err_t


def compute_maa(err_q, err_t, thresholds_q=None, thresholds_t=None):
    thresholds_q = np.linspace(1, 10, 10) if thresholds_q is None else thresholds_q
    thresholds_t = np.geomspace(0.2, 5, 10) if thresholds_t is None else thresholds_t
    err_q = np.asarray(err_q)
    err_t = np.asarray(err_t)
    acc = [((err_q < tq) & (err_t < tt)).mean() for tq, tt in zip(thresholds_q, thresholds_t)]
    return float(np.mean(acc))


def estimate_pose_from_matches(kpts_a, kpts_b, K_a, K_b, threshold=0.25):
    if len(kpts_a) < 8:
        return None
    F, mask = cv2.findFundamentalMat(
        kpts_a,
        kpts_b,
        cv2.USAC_MAGSAC,
        threshold,
        0.999999,
        20000,
    )
    if F is None or mask is None:
        return None
    mask = mask.astype(bool).ravel()
    in_a, in_b = kpts_a[mask], kpts_b[mask]
    if len(in_a) < 8:
        return None
    E, R, T = compute_essential_matrix(F, K_a, K_b, in_a, in_b)
    return F, E, R, T.flatten(), mask, in_a, in_b


def relative_ground_truth(calibration, id1, id2):
    R1 = calibration[id1].R
    T1 = calibration[id1].T.reshape(3, 1)
    R2 = calibration[id2].R
    T2 = calibration[id2].T.reshape(3, 1)
    dR = R2 @ R1.T
    dT = (T2 - dR @ T1).flatten()
    q = quaternion_from_matrix(dR)
    q /= np.linalg.norm(q) + EPS
    return q, dT
