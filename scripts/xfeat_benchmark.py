import os
import sys
import time
import cv2
import numpy as np
import torch

# =========================
# PATHS
# =========================
BASE_DIR = os.path.expanduser("~/feature_matching_task")
XFEAT_REPO = os.path.join(BASE_DIR, "repos", "accelerated_features")
sys.path.append(XFEAT_REPO)

IMG1_PATH = os.path.join(BASE_DIR, "data", "tgt.png")
IMG2_PATH = os.path.join(BASE_DIR, "data", "ref.png")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULT_DIR, "xfeat_matches_v2.png")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =========================
# IMPORT
# =========================
from modules.xfeat import XFeat

# =========================
# SETTINGS
# =========================
TOP_K = 200
RESIZE_SCALE = 0.75
WARMUP = 2
RUNS = 10
TOP_DRAW = 100
RANSAC_THRESH = 3.0

# =========================
# LOAD IMAGES
# =========================
img1_bgr = cv2.imread(IMG1_PATH)
img2_bgr = cv2.imread(IMG2_PATH)

if img1_bgr is None:
    raise FileNotFoundError(f"img1 okunamadı: {IMG1_PATH}")
if img2_bgr is None:
    raise FileNotFoundError(f"img2 okunamadı: {IMG2_PATH}")

if RESIZE_SCALE != 1.0:
    img1_bgr = cv2.resize(img1_bgr, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    img2_bgr = cv2.resize(img2_bgr, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)

img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

x1 = torch.from_numpy(img1_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
x2 = torch.from_numpy(img2_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# =========================
# MODEL
# =========================
xfeat = XFeat()

extract_times = []
match_times = []
total_times = []

last_out1 = None
last_out2 = None
last_mkpts0 = None
last_mkpts1 = None

def to_cv_keypoints(kps):
    out = []
    for p in kps:
        out.append(cv2.KeyPoint(float(p[0]), float(p[1]), 1))
    return out

def compute_ransac_metrics(mkpts0, mkpts1, ransac_thresh=3.0):
    if len(mkpts0) < 4 or len(mkpts1) < 4:
        return {
            "homography_found": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": None,
            "H": None,
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
            "H": None,
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
        "H": H,
        "mask": mask,
    }

def nearest_indices(points, query_points):
    """
    query_points içindeki her nokta için points içindeki en yakın indeks.
    """
    idxs = []
    for q in query_points:
        idx = int(np.argmin(np.linalg.norm(points - q, axis=1)))
        idxs.append(idx)
    return np.array(idxs, dtype=np.int32)

with torch.no_grad():
    for _ in range(WARMUP):
        _ = xfeat.detectAndCompute(x1, top_k=TOP_K)[0]
        _ = xfeat.detectAndCompute(x2, top_k=TOP_K)[0]
        _ = xfeat.match_xfeat(x1, x2)

    for _ in range(RUNS):
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        out1 = xfeat.detectAndCompute(x1, top_k=TOP_K)[0]
        out2 = xfeat.detectAndCompute(x2, top_k=TOP_K)[0]
        t2 = time.perf_counter()

        mkpts0, mkpts1 = xfeat.match_xfeat(x1, x2)
        t3 = time.perf_counter()

        extract_times.append((t2 - t1) * 1000.0)
        match_times.append((t3 - t2) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)

        last_out1 = out1
        last_out2 = out2
        last_mkpts0 = mkpts0
        last_mkpts1 = mkpts1

if last_out1 is None or last_out2 is None:
    raise RuntimeError("XFeat çıktı üretmedi.")

kpts1 = last_out1["keypoints"].cpu().numpy()
kpts2 = last_out2["keypoints"].cpu().numpy()
scores1 = last_out1["scores"].cpu().numpy()
scores2 = last_out2["scores"].cpu().numpy()

mkpts0 = last_mkpts0.cpu().numpy() if torch.is_tensor(last_mkpts0) else np.asarray(last_mkpts0)
mkpts1 = last_mkpts1.cpu().numpy() if torch.is_tensor(last_mkpts1) else np.asarray(last_mkpts1)

# =========================
# APPROX CONFIDENCE
# =========================
if len(mkpts0) > 0 and len(kpts1) > 0 and len(kpts2) > 0:
    idx1 = nearest_indices(kpts1, mkpts0)
    idx2 = nearest_indices(kpts2, mkpts1)
    match_conf = (scores1[idx1] + scores2[idx2]) / 2.0
    avg_conf = float(np.mean(match_conf))
    min_conf = float(np.min(match_conf))
    max_conf = float(np.max(match_conf))
else:
    idx1 = np.array([], dtype=np.int32)
    idx2 = np.array([], dtype=np.int32)
    match_conf = np.array([], dtype=np.float32)
    avg_conf = 0.0
    min_conf = 0.0
    max_conf = 0.0

# =========================
# QUALITY / ACCURACY
# =========================
quality = compute_ransac_metrics(mkpts0, mkpts1, ransac_thresh=RANSAC_THRESH)
accuracy_like = quality["inlier_ratio"]

# =========================
# SPEED METRICS
# =========================
avg_extract = float(np.mean(extract_times))
avg_match = float(np.mean(match_times))
avg_total = float(np.mean(total_times))
min_total = float(np.min(total_times))
max_total = float(np.max(total_times))
hz = 1000.0 / avg_total if avg_total > 0 else 0.0

# =========================
# DRAW MATCHES
# =========================
cv_kp1 = to_cv_keypoints(kpts1)
cv_kp2 = to_cv_keypoints(kpts2)

draw_matches = []
if len(mkpts0) > 0 and len(kpts1) > 0 and len(kpts2) > 0:
    if quality["mask"] is not None:
        inlier_indices = np.where(quality["mask"])[0]
    else:
        inlier_indices = np.arange(len(mkpts0))

    inlier_indices = inlier_indices[:TOP_DRAW]

    for idx in inlier_indices:
        i = int(idx1[idx])
        j = int(idx2[idx])
        draw_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0))

vis = cv2.drawMatches(
    cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY), cv_kp1,
    cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY), cv_kp2,
    draw_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(OUT_PATH, vis)

# =========================
# PRINT
# =========================
print("=" * 60)
print("XFEAT BENCHMARK V2 RESULT")
print("=" * 60)
print(f"resize scale              : {RESIZE_SCALE}")
print(f"top_k                     : {TOP_K}")
print(f"img1 keypoint sayisi      : {len(kpts1)}")
print(f"img2 keypoint sayisi      : {len(kpts2)}")
print(f"toplam match sayisi       : {len(mkpts0)}")
print(f"gosterilen inlier match   : {len(draw_matches)}")
print(f"min confidence            : {min_conf:.3f}")
print(f"max confidence            : {max_conf:.3f}")
print(f"ortalama confidence       : {avg_conf:.3f}")
print("-" * 60)
print(f"homography bulundu mu     : {quality['homography_found']}")
print(f"inlier sayisi             : {quality['inliers']}")
print(f"inlier ratio (accuracy)   : {accuracy_like:.4f}")
print(f"reprojection error        : {quality['reproj_error']}")
print("-" * 60)
print(f"ortalama extraction       : {avg_extract:.2f} ms")
print(f"ortalama matching         : {avg_match:.2f} ms")
print(f"ortalama toplam sure      : {avg_total:.2f} ms")
print(f"min toplam sure           : {min_total:.2f} ms")
print(f"max toplam sure           : {max_total:.2f} ms")
print(f"ortalama frekans          : {hz:.2f} Hz")
print(f"sonuc dosyasi             : {OUT_PATH}")
print("=" * 60)