import os
import sys
import time
import cv2
import numpy as np

# =========================
# PATHS
# =========================
BASE_DIR = os.path.expanduser("~/feature_matching_task")
ALIKE_REPO = os.path.join(BASE_DIR, "repos", "ALIKE")
sys.path.append(ALIKE_REPO)

IMG1_PATH = ("/home/merve/feature_matching_task/data/000100.png")
IMG2_PATH = ("/home/merve/feature_matching_task/data/000108.png")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULT_DIR, "alike_matches_optimized.png")

# =========================
# IMPORT
# =========================
from alike import ALike, configs

# =========================
# LOAD IMAGES
# =========================
img1_bgr = cv2.imread(IMG1_PATH)
img2_bgr = cv2.imread(IMG2_PATH)

if img1_bgr is None:
    raise FileNotFoundError(f"img1 okunamadı: {IMG1_PATH}")
if img2_bgr is None:
    raise FileNotFoundError(f"img2 okunamadı: {IMG2_PATH}")

# =========================
# OPTIMIZATION SETTINGS
# =========================
RESIZE_SCALE = 0.75
MODEL_NAME = "alike-t"
DEVICE = "cpu"
TOP_K = -1
SCORES_TH = 0.35
N_LIMIT = 200
SIM_TH = 0.90
WARMUP = 2
RUNS = 10
TOP_DRAW = 100
RANSAC_THRESH = 3.0

if RESIZE_SCALE != 1.0:
    img1_bgr = cv2.resize(img1_bgr, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    img2_bgr = cv2.resize(img2_bgr, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)

# ALIKE RGB bekliyor
img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# MODEL
# =========================
model = ALike(
    **configs[MODEL_NAME],
    device=DEVICE,
    top_k=TOP_K,
    scores_th=SCORES_TH,
    n_limit=N_LIMIT
)

# =========================
# HELPERS
# =========================
def to_cv_keypoints(kps):
    return [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps]

def safe_len(x):
    return 0 if x is None else len(x)

def mnn_matcher(desc1, desc2, sim_th=0.9):
    if desc1 is None or desc2 is None:
        return np.zeros((0, 2), dtype=np.int32)
    if len(desc1) == 0 or len(desc2) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    sim = desc1 @ desc2.T
    sim[sim < sim_th] = 0

    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)

    ids1 = np.arange(sim.shape[0])
    mask = (ids1 == nn21[nn12])

    matches = np.stack([ids1[mask], nn12[mask]], axis=1)
    return matches.astype(np.int32)

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

# =========================
# WARMUP
# =========================
for _ in range(WARMUP):
    pred1 = model(img1)
    pred2 = model(img2)
    _ = mnn_matcher(pred1["descriptors"], pred2["descriptors"], sim_th=SIM_TH)

# =========================
# BENCHMARK
# =========================
extract_times = []
match_times = []
total_times = []

last_pred1 = None
last_pred2 = None
last_matches = None

for _ in range(RUNS):
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    pred1 = model(img1)
    pred2 = model(img2)
    t2 = time.perf_counter()

    matches = mnn_matcher(pred1["descriptors"], pred2["descriptors"], sim_th=SIM_TH)
    t3 = time.perf_counter()

    extract_times.append((t2 - t1) * 1000.0)
    match_times.append((t3 - t2) * 1000.0)
    total_times.append((t3 - t0) * 1000.0)

    last_pred1 = pred1
    last_pred2 = pred2
    last_matches = matches

# =========================
# CHECK
# =========================
if last_pred1 is None or last_pred2 is None or last_matches is None:
    raise RuntimeError("Benchmark sırasında çıktı üretilemedi.")

kpts1 = last_pred1["keypoints"]
kpts2 = last_pred2["keypoints"]
scores1 = last_pred1["scores"]
scores2 = last_pred2["scores"]

if len(last_matches) == 0:
    idx1 = np.array([], dtype=np.int32)
    idx2 = np.array([], dtype=np.int32)
    match_conf = np.array([], dtype=np.float32)
    mkpts0 = np.empty((0, 2), dtype=np.float32)
    mkpts1 = np.empty((0, 2), dtype=np.float32)
else:
    idx1 = last_matches[:, 0].astype(int)
    idx2 = last_matches[:, 1].astype(int)
    mkpts0 = kpts1[idx1]
    mkpts1 = kpts2[idx2]
    match_conf = (scores1[idx1] + scores2[idx2]) / 2.0

# =========================
# QUALITY
# =========================
quality = compute_ransac_metrics(mkpts0, mkpts1, ransac_thresh=RANSAC_THRESH)

# =========================
# METRICS
# =========================
avg_extract = float(np.mean(extract_times))
avg_match = float(np.mean(match_times))
avg_total = float(np.mean(total_times))
min_total = float(np.min(total_times))
max_total = float(np.max(total_times))
hz = 1000.0 / avg_total if avg_total > 0 else 0.0

min_conf = float(np.min(match_conf)) if len(match_conf) else 0.0
max_conf = float(np.max(match_conf)) if len(match_conf) else 0.0
avg_conf = float(np.mean(match_conf)) if len(match_conf) else 0.0

# =========================
# DRAW
# =========================
cv_kp1 = to_cv_keypoints(kpts1)
cv_kp2 = to_cv_keypoints(kpts2)

draw_matches = []
if len(last_matches) > 0:
    if quality["mask"] is not None:
        inlier_indices = np.where(quality["mask"])[0]
    else:
        inlier_indices = np.arange(len(last_matches))

    inlier_indices = inlier_indices[:TOP_DRAW]

    for idx in inlier_indices:
        a = int(idx1[idx])
        b = int(idx2[idx])
        draw_matches.append(cv2.DMatch(_queryIdx=a, _trainIdx=b, _distance=0))

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
print("ALIKE OPTIMIZED BENCHMARK RESULT")
print("=" * 60)
print(f"resize scale              : {RESIZE_SCALE}")
print(f"model                     : {MODEL_NAME}")
print(f"scores_th                 : {SCORES_TH}")
print(f"n_limit                   : {N_LIMIT}")
print(f"img1 keypoint sayisi      : {safe_len(kpts1)}")
print(f"img2 keypoint sayisi      : {safe_len(kpts2)}")
print(f"toplam match sayisi       : {safe_len(last_matches)}")
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
print(f"min toplam sure           : {min_total:.2f} ms")
print(f"max toplam sure           : {max_total:.2f} ms")
print(f"ortalama frekans          : {hz:.2f} Hz")
print(f"sonuc dosyasi             : {OUT_PATH}")
print("=" * 60)