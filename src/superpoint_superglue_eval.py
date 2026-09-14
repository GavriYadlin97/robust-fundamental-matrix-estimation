import os
import numpy as np
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import torch

# =========== NEW: Import Matching ===========
from models.matching import Matching

random.seed(37)

# A named tuple containing the intrinsics (calibration matrix K) and extrinsics (rotation matrix R, translation vector T)
Gt = namedtuple('Gt', ['K', 'R', 'T'])

# A small epsilon.
eps = 1e-15

# -------------------------------
# Helper function: pad image so that height and width are divisible by a given divisor.
def pad_to_divisor(img, divisor=8):
    """Pads the image (numpy array) so that its height and width are divisible by 'divisor'."""
    h, w = img.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if len(img.shape) == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    elif len(img.shape) == 3:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        padded = img
    return padded, pad_h, pad_w

def ReadCovisibilityData(filename):
    covisibility_dict = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            covisibility_dict[row[1]] = float(row[2])
    return covisibility_dict

def NormalizeKeypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def ComputeEssentialMatrix(F, K1, K2, kp1, kp2):
    assert F.shape[0] == 3, 'Malformed F?'
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)
    kp1n = NormalizeKeypoints(kp1, K1)
    kp2n = NormalizeKeypoints(kp2, K2)
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)
    return E, R, T

def QuaternionFromMatrix(matrix):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    K_mat = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K_mat /= 3.0
    w, V = np.linalg.eigh(K_mat)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0:
        np.negative(q, q)
    return q

def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)
    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))
    return err_q * 180 / np.pi, err_t

def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    assert len(err_q) == len(err_t)
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)

def BuildCompositeImage(im1, im2, axis=1, margin=0, background=0):
    """
    Build a composite image of im1 and im2. Here, we set margin=0 and background=0 (black)
    to avoid extra unwanted white areas.
    """
    if axis not in [0, 1]:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal)')
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8)
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3), dtype=np.uint8)
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2
    return composite, (voff1, voff2), (hoff1, hoff2)

def crop_composite(composite, im1, im2, voff1, voff2, hoff1, hoff2):
    """
    Crop the composite image to tightly bound the two images.
    """
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    top = min(voff1, voff2)
    left = min(hoff1, hoff2)
    bottom = max(voff1 + h1, voff2 + h2)
    right = max(hoff1 + w1, hoff2 + w2)
    return composite[top:bottom, left:right]

def DrawMatches(im1, im2, kp1, kp2, matches, axis=1, margin=0, background=0, linewidth=2):
    # Build the composite image with the given parameters.
    composite, (voff1, voff2), (hoff1, hoff2) = BuildCompositeImage(im1, im2, axis, margin, background)
    # Crop the composite image to remove extra background area.
    composite = crop_composite(composite, im1, im2, voff1, voff2, hoff1, hoff2)
    # Draw keypoints and matches.
    for coord_a, coord_b in zip(kp1, kp2):
        composite = cv2.drawMarker(composite,
                                   (int(coord_a[0] + hoff1), int(coord_a[1] + voff1)),
                                   color=(255, 0, 0),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=5,
                                   thickness=1)
        composite = cv2.drawMarker(composite,
                                   (int(coord_b[0] + hoff2), int(coord_b[1] + voff2)),
                                   color=(255, 0, 0),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=5,
                                   thickness=1)
    for idx_a, idx_b in matches:
        composite = cv2.drawMarker(composite,
                                   (int(kp1[idx_a][0] + hoff1), int(kp1[idx_a][1] + voff1)),
                                   color=(0, 0, 255),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=12,
                                   thickness=1)
        composite = cv2.drawMarker(composite,
                                   (int(kp2[idx_b][0] + hoff2), int(kp2[idx_b][1] + voff2)),
                                   color=(0, 0, 255),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=12,
                                   thickness=1)
        composite = cv2.line(composite,
                             (int(kp1[idx_a][0] + hoff1), int(kp1[idx_a][1] + voff1)),
                             (int(kp2[idx_b][0] + hoff2), int(kp2[idx_b][1] + voff2)),
                             color=(0, 0, 255),
                             thickness=linewidth)
    return composite

def LoadCalibration(filename):
    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            camera_id = row[1]
            K = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[3].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[4].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)
    return calib_dict

# ======================================================================
# ======================= MAIN SCRIPT (Matching over All Scenes) =========
# ======================================================================
if __name__ == "__main__":
    # 1) Initialize Matching (SuperPoint + SuperGlue)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'superpoint': {
            'nms_radius': 2,
            'keypoint_threshold': 0.004,
            'max_keypoints': 4096,
            'remove_borders': 2,
        },
        'superglue': {
            'weights': 'outdoor',  # use outdoor weights for building images
            'sinkhorn_iterations': 100,  # increased iterations
            'match_threshold': 0.05,  # lower threshold for more selective matching
        }
    }
    matching = Matching(config).eval().to(device)

    # 2) Prepare dataset
    src = "Data/train"
    val_scenes = []
    for f in os.scandir(src):
        if f.is_dir():
            scene_name = os.path.split(f)[-1]
            print(f'Found scene "{scene_name}" at {f.path}')
            val_scenes.append(scene_name)

    # Load scaling factors (assumed global for the dataset)
    scaling_dict = {}
    with open(f"{src}/scaling_factors.csv") as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            scaling_dict[row[1]] = float(row[2])
    print(f"Scaling factors: {scaling_dict}\n")

    # To accumulate overall mAA across scenes:
    scene_mAA = {}

    # Loop over all scenes
    for scene in val_scenes:
        print(f"\n======= Processing scene: {scene} =======")
        # Load images for this scene
        images_dict = {}
        image_files = glob(f"{src}/{scene}/images/*.jpg")
        for filename in image_files:
            cur_id = os.path.basename(os.path.splitext(filename)[0])
            images_dict[cur_id] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        print(f"Loaded {len(images_dict)} images for scene '{scene}'.")

        # Load pair covisibility data for the scene
        covisibility_dict = ReadCovisibilityData(f"{src}/{scene}/pair_covisibility.csv")
        # Filter pairs (here using a threshold of 0.0)
        pairs = [p for p, covis in covisibility_dict.items() if covis >= 0.0]
        max_pairs_per_scene = 10
        print(f"Found {len(pairs)} pairs in scene '{scene}'. Processing up to {max_pairs_per_scene} pairs.")
        random.shuffle(pairs)
        pairs = pairs[:max_pairs_per_scene]

        # Collect all image ids used in pairs
        ids = []
        for pair in pairs:
            cur_ids = pair.split('-')
            ids.extend(cur_ids)
        ids = list(set(ids))

        # Load calibration for the scene
        calib_dict_scene = LoadCalibration(f"{src}/{scene}/calibration.csv")

        # Reload images (if needed) for the loop over pairs
        images_dict_scene = {}
        for cid in ids:
            imgBGR = cv2.imread(f"{src}/{scene}/images/{cid}.jpg")
            images_dict_scene[cid] = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        print(f"Loaded {len(images_dict_scene)} images for pairs processing.")

        # To store errors for each pair
        errors_scene = {}

        # Loop over pairs in the scene
        for counter, pair in enumerate(pairs):
            id1, id2 = pair.split('-')
            img0 = images_dict_scene[id1]
            img1 = images_dict_scene[id2]

            # Convert images to grayscale and pad
            gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            gray0, _, _ = pad_to_divisor(gray0, divisor=8)
            gray1, _, _ = pad_to_divisor(gray1, divisor=8)

            tensor0 = torch.from_numpy(gray0)[None, None, ...].to(device)
            tensor1 = torch.from_numpy(gray1)[None, None, ...].to(device)
            data = {"image0": tensor0, "image1": tensor1}

            with torch.no_grad():
                pred = matching(data)

            # Obtain keypoints and match indices from Matching
            kp0 = pred['keypoints0'][0].cpu().numpy()  # shape [N,2]
            kp1 = pred['keypoints1'][0].cpu().numpy()  # shape [M,2]
            matches0 = pred['matches0'][0].cpu().numpy()  # shape [N,] (-1 indicates no match)

            # Filter out unmatched keypoints (where matches0 == -1)
            valid = matches0 > -1
            mkpts0 = kp0[valid]
            mkpts1 = kp1[matches0[valid]]
            if mkpts0.shape[0] < 8:
                print(f"Pair {pair} skipped: not enough matching pairs ({mkpts0.shape[0]})")
                continue

            # Compute the Fundamental matrix using the matched keypoints
            F, inlier_mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.25, 0.999, 20000)
            if inlier_mask is None or inlier_mask.size == 0:
                print(f"Pair {pair} skipped: no inliers found.")
                continue

            inlier_mask = inlier_mask.astype(bool).flatten()
            inlier_mkpts0 = mkpts0[inlier_mask]
            inlier_mkpts1 = mkpts1[inlier_mask]
            if inlier_mkpts0.shape[0] < 8:
                print(f"Pair {pair} skipped: not enough inlier matches ({inlier_mkpts0.shape[0]})")
                continue

            # Compute E, recover pose, and compute errors using inlier matches
            E, R, T = ComputeEssentialMatrix(
                F,
                calib_dict_scene[id1].K,
                calib_dict_scene[id2].K,
                inlier_mkpts0,
                inlier_mkpts1
            )
            q = QuaternionFromMatrix(R)
            T = T.flatten()

            R1_gt, T1_gt = calib_dict_scene[id1].R, calib_dict_scene[id1].T.reshape((3, 1))
            R2_gt, T2_gt = calib_dict_scene[id2].R, calib_dict_scene[id2].T.reshape((3, 1))
            dR_gt = np.dot(R2_gt, R1_gt.T)
            dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
            q_gt = QuaternionFromMatrix(dR_gt)
            q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

            err_q, err_t = ComputeErrorForOneExample(
                q_gt, dT_gt, q, T, scaling_dict[scene]
            )
            errors_scene[pair] = [err_q, err_t]

            # Visualize the inlier matches for the first pair in this scene
            if counter < 1:
                # Create a simple index pairing for the inlier matches
                inlier_indices = np.arange(len(inlier_mkpts0))
                matches_after_ransac = np.stack([inlier_indices, inlier_indices], axis=1)
                im_inliers = DrawMatches(img0, img1, inlier_mkpts0, inlier_mkpts1, matches_after_ransac,
                                         margin=0, background=0)
                fig = plt.figure(figsize=(25, 25))
                plt.title(f'Inliers for pair "{pair}" in scene "{scene}"')
                plt.imshow(im_inliers)
                plt.axis('off')
                plt.show()

            print(f'Pair {pair}, rotation error: {err_q:.02f} deg, translation error: {err_t:.02f} m')

        # Compute mAA for this scene if there are valid pairs
        if len(errors_scene) > 0:
            err_qs = [v[0] for v in errors_scene.values()]
            err_ts = [v[1] for v in errors_scene.values()]
            mAA_scene = ComputeMaa(err_qs, err_ts, np.linspace(1, 10, 10), np.geomspace(0.2, 5, 10))
            mAA_value = mAA_scene[0]
            scene_mAA[scene] = mAA_value
            print(f'\nMean average Accuracy on scene "{scene}": {mAA_value:.05f}\n')
        else:
            scene_mAA[scene] = 0
            print(f'\nNo valid pairs for scene "{scene}". mAA=0.\n')

    # Summary over all scenes
    print('\n------- SUMMARY OVER ALL SCENES -------\n')
    for sc, acc in scene_mAA.items():
        print(f'Scene "{sc}": Mean average Accuracy = {acc:.05f}')
    overall_mAA = np.mean(list(scene_mAA.values()))
    print(f'\nOverall Mean average Accuracy on dataset: {overall_mAA:.05f}')
