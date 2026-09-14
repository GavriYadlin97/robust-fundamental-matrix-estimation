# Project Report Summary

This repository is based on a project for **Introduction to Computer Vision (22928)** at the Open University of Israel. The original project was framed as a Kaggle competition: estimate the fundamental matrix between two views of the same scene despite changes in illumination, weather, viewpoint, and partial occlusion.

## Development path

The project evolved through several increasingly modern matching approaches:

1. **Classical local features (SIFT)** — established an initial baseline around 0.30 mAA.
2. **Direct deep regression attempts** — features from architectures such as ResNet, VGG, and ViT were explored as inputs to a network predicting the flattened 3×3 fundamental matrix. These experiments were abandoned because performance was poor and training was computationally expensive.
3. **SuperPoint + SuperGlue** — learned feature detection/description and graph-attention-based matching improved the reported score to roughly 0.45–0.50 mAA after parameter tuning and MAGSAC tuning.
4. **LoFTR** — detector-free transformer matching further improved the reported score to roughly 0.62 mAA.
5. **RoMa** — dense robust matching achieved the strongest completed result, approximately 0.68 mAA, but required substantially more compute.
6. **Ensemble exploration** — candidate matrices were estimated from LoFTR matches, RoMa matches, and their concatenation. An oracle-style selector based on ground-truth pose errors reached roughly 0.77 mAA on a small validation subset, demonstrating potential but not providing a deployable test-time selection method.

## Geometric estimation

For the learned matching approaches, corresponding points are passed to OpenCV's **USAC-MAGSAC** estimator to recover the fundamental matrix. On validation data, the fundamental matrix is converted to an essential matrix using camera intrinsics, relative pose is recovered, and rotation/translation errors are computed against calibration ground truth.

## Evaluation

The principal metric is **mean Average Accuracy (mAA)** over rotation and translation error thresholds. The project therefore evaluates not only whether image matches appear plausible, but whether they support accurate relative camera geometry.

## Main takeaway

The largest gains came from replacing classical or generic image representations with matching models explicitly designed for geometric correspondence. Dense transformer-based matching (RoMa) produced the best completed submission, while the ensemble experiment suggested that different matching models can be complementary on different image pairs.
