# Robust Fundamental Matrix Estimation

Computer-vision project for estimating the **fundamental matrix** between two views of the same scene under challenging real-world conditions such as viewpoint, lighting, weather, and partial occlusion changes.

The project compares classical and learned image-matching approaches and uses **USAC-MAGSAC** for robust geometric estimation.

## Highlights

- Evaluated **SIFT**, **SuperPoint + SuperGlue**, **LoFTR**, and **RoMa** for image correspondence.
- Estimated the fundamental matrix with OpenCV's **USAC-MAGSAC** robust estimator.
- Recovered relative camera pose from the estimated geometry for validation.
- Measured performance with **mean Average Accuracy (mAA)** using rotation and translation errors.
- Explored an ensemble strategy combining LoFTR and RoMa correspondences.
- Built a resumable RoMa inference pipeline for generating the final competition submission.

## Reported project results

| Method | Approx. mAA | Notes |
| --- | ---: | --- |
| SIFT + robust estimation | ~0.30 | Classical baseline explored early in the project |
| SuperPoint + SuperGlue + MAGSAC | ~0.50 | Learned sparse keypoints and matching |
| LoFTR + MAGSAC | ~0.62 | Transformer-based detector-free matching |
| **RoMa + MAGSAC** | **~0.68** | Best completed submission pipeline |
| LoFTR / RoMa ensemble experiment | ~0.77* | Reduced validation subset only; not used in final submission |

\* The ensemble result was obtained on a small validation subset (about 10 pairs per scene), so it should not be interpreted as directly comparable to the full test-set score.

## Pipeline

```text
Image pair
   │
   ├── SuperPoint + SuperGlue
   ├── LoFTR
   └── RoMa
          │
          ▼
Matched pixel correspondences
          │
          ▼
USAC-MAGSAC robust estimation
          │
          ▼
Fundamental matrix F
          │
          ├── Essential matrix / pose recovery (validation)
          └── Kaggle submission output (test)
```

## Repository structure

```text
.
├── src/
│   ├── superpoint_superglue_eval.py  # SuperPoint + SuperGlue evaluation
│   ├── loftr_eval.py                 # LoFTR evaluation
│   ├── roma_eval.py                  # RoMa evaluation and visualizations
│   └── roma_submission.py            # Final RoMa submission generator
├── experiments/
│   └── ensemble_selection.py         # Experimental LoFTR/RoMa combination
├── requirements.txt
└── .gitignore
```

## Methods

### SuperPoint + SuperGlue

SuperPoint detects and describes image keypoints with a learned model, while SuperGlue performs context-aware feature matching. The matched correspondences are filtered by USAC-MAGSAC before estimating the fundamental matrix.

### LoFTR

LoFTR is a detector-free transformer matcher. Instead of detecting sparse interest points first, it directly establishes correspondences between image regions using coarse-to-fine matching.

### RoMa

RoMa produces dense robust correspondences and performed best among the completed approaches in this project. Its larger number of matches improved geometric estimation under difficult viewpoint changes, at the cost of substantially higher compute time.

### Ensemble experiment

The exploratory ensemble computes candidate matrices from:

1. LoFTR matches,
2. RoMa matches, and
3. the concatenation of both match sets.

The experiment then compares candidate pose errors when ground truth is available. This was a proof of concept for a learned or correspondence-based matrix selector and was **not part of the final competition submission**.

## Setup

The scripts depend on upstream implementations of the matching models rather than vendoring those repositories here.

Clone the required model repositories into the project root (or adjust the imports/paths in the scripts):

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
git clone https://github.com/zju3dv/LoFTR.git
git clone https://github.com/Parskatt/RoMa.git
```

Install the common Python dependencies:

```bash
pip install -r requirements.txt
```

The scripts expect the competition data under `Data/` and the LoFTR outdoor checkpoint under `LoFTR/weights/` unless the paths are edited.

## Running

Examples from the repository root:

```bash
python src/superpoint_superglue_eval.py
python src/loftr_eval.py
python src/roma_eval.py
python src/roma_submission.py
```

The evaluation scripts sample image pairs from each scene, estimate camera geometry, and report scene-level and overall mAA. The submission script processes test pairs and appends estimated matrices to a CSV so runs can be resumed.

## Skills demonstrated

`Python` · `PyTorch` · `OpenCV` · `Computer Vision` · `Epipolar Geometry` · `Feature Matching` · `Transformers` · `Robust Estimation` · `RANSAC / MAGSAC` · `Experimental Evaluation`

## Project context

Developed for the Open University of Israel course **Introduction to Computer Vision (22928)** as a Kaggle-style project on robust fundamental-matrix estimation.

Third-party model implementations and pretrained weights remain the property of their respective authors and are not redistributed in this repository.