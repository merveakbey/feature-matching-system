import os
import sys
import time
import cv2
import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/feature_matching_task")
LIGHTGLUE_REPO = os.path.join(BASE_DIR, "repos", "LightGlue")
sys.path.append(LIGHTGLUE_REPO)

IMG1_PATH = os.path.join(BASE_DIR, "repos", "LightGlue", "assets", "sacre_coeur1.jpg")
IMG2_PATH = os.path.join(BASE_DIR, "repos", "LightGlue", "assets", "sacre_coeur2.jpg")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULT_DIR, "aliked_lightglue_matches.png")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd

TOP_K = 200
RESIZE = 768
WARMUP = 2
RUNS = 10
RANSAC_THRESH = 3.0
TOP_DRAW = 100

extractor = ALIKED(max_num_keypoints=TOP_K).eval()
matcher = LightGlue(features="aliked").eval()

image0 = load_image(IMG1_PATH)
image1 = load_image(IMG2_PATH)

extract_times = []
match_times = []
total_times = []

last_feats0 = None
last_feats1 = None
last_matches01 = None

def compute_ransac_metrics(mkpts0, mkpts1, ransac_thresh=3.0):
    if len(mkpts0) < 4 or len(mkpts1) < 4:
        return {
            "homography_found": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": None,
            "mask": None,
        }

    H, mask = cv2.findHomography(
        mkpts0.reshape(-1, 1, 2),
        mkpts1.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_thresh
    )

    if H is None or mask is None:
        return {
            "homography_found": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": None,
            "mask": None,
        }

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    total = len(mask)
    inlier_ratio = inliers / total if total > 0 else 0.0

    pts0_h = cv2.convertPointsToHomogeneous(mkpts0).reshape(-1, 3)
    proj = (H @ pts0_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj - mkpts1, axis=1)
    reproj_error = float(np.mean(errors[mask])) if inliers > 0 else None

    return {
        "homography_found": True,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "reproj_error": reproj_error,
        "mask": mask,
    }

with torch.no_grad():
    for _ in range(WARMUP):
        feats0 = extractor.extract(image0, resize=RESIZE)
        feats1 = extractor.extract(image1, resize=RESIZE)
        _ = matcher({"image0": feats0, "image1": feats1})

    for _ in range(RUNS):
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        feats0 = extractor.extract(image0, resize=RESIZE)
        feats1 = extractor.extract(image1, resize=RESIZE)
        t2 = time.perf_counter()

        matches01 = matcher({"image0": feats0, "image1": feats1})
        t3 = time.perf_counter()

        extract_times.append((t2 - t1) * 1000.0)
        match_times.append((t3 - t2) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)

        last_feats0 = feats0
        last_feats1 = feats1
        last_matches01 = matches01

feats0, feats1, matches01 = [rbd(x) for x in [last_feats0, last_feats1, last_matches01]]

matches = matches01["matches"]
scores = matches01["scores"] if "scores" in matches01 else None

kpts0 = feats0["keypoints"].cpu().numpy() if torch.is_tensor(feats0["keypoints"]) else np.asarray(feats0["keypoints"])
kpts1 = feats1["keypoints"].cpu().numpy() if torch.is_tensor(feats1["keypoints"]) else np.asarray(feats1["keypoints"])

matches_np = matches.cpu().numpy() if torch.is_tensor(matches) else np.asarray(matches)
scores_np = scores.cpu().numpy() if (scores is not None and torch.is_tensor(scores)) else (np.asarray(scores) if scores is not None else np.array([]))

mkpts0 = kpts0[matches_np[:, 0]] if len(matches_np) else np.empty((0, 2), dtype=np.float32)
mkpts1 = kpts1[matches_np[:, 1]] if len(matches_np) else np.empty((0, 2), dtype=np.float32)

quality = compute_ransac_metrics(mkpts0, mkpts1, ransac_thresh=RANSAC_THRESH)

avg_extract = float(np.mean(extract_times))
avg_match = float(np.mean(match_times))
avg_total = float(np.mean(total_times))
hz = 1000.0 / avg_total if avg_total > 0 else 0.0

avg_conf = float(np.mean(scores_np)) if len(scores_np) else 0.0
min_conf = float(np.min(scores_np)) if len(scores_np) else 0.0
max_conf = float(np.max(scores_np)) if len(scores_np) else 0.0

img0_gray = cv2.imread(IMG1_PATH, cv2.IMREAD_GRAYSCALE)
img1_gray = cv2.imread(IMG2_PATH, cv2.IMREAD_GRAYSCALE)

cv_kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kpts0]
cv_kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kpts1]

draw_matches = []
if len(matches_np):
    idxs = np.where(quality["mask"])[0] if quality["mask"] is not None else np.arange(len(matches_np))
    idxs = idxs[:TOP_DRAW]
    for idx in idxs:
        a = int(matches_np[idx, 0])
        b = int(matches_np[idx, 1])
        draw_matches.append(cv2.DMatch(_queryIdx=a, _trainIdx=b, _distance=0))

vis = cv2.drawMatches(
    img0_gray, cv_kp0,
    img1_gray, cv_kp1,
    draw_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(OUT_PATH, vis)

print("=" * 60)
print("ALIKED + LIGHTGLUE RESULT")
print("=" * 60)
print(f"top_k                     : {TOP_K}")
print(f"resize                    : {RESIZE}")
print(f"img1 keypoint sayisi      : {len(kpts0)}")
print(f"img2 keypoint sayisi      : {len(kpts1)}")
print(f"toplam match sayisi       : {len(matches_np)}")
print(f"gosterilen inlier match   : {len(draw_matches)}")
print(f"min confidence            : {min_conf:.3f}")
print(f"max confidence            : {max_conf:.3f}")
print(f"ortalama confidence       : {avg_conf:.3f}")
print("-" * 60)
print(f"homography bulundu mu     : {quality['homography_found']}")
print(f"inlier sayisi             : {quality['inliers']}")
print(f"inlier ratio (accuracy)   : {quality['inlier_ratio']:.4f}")
print(f"reprojection error        : {quality['reproj_error']}")
print("-" * 60)
print(f"ortalama extraction       : {avg_extract:.2f} ms")
print(f"ortalama matching         : {avg_match:.2f} ms")
print(f"ortalama toplam sure      : {avg_total:.2f} ms")
print(f"ortalama frekans          : {hz:.2f} Hz")
print(f"sonuc dosyasi             : {OUT_PATH}")
print("=" * 60)