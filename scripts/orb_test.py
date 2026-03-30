import cv2
import time
import os
import numpy as np

# =========================
# PATHS
# =========================
BASE_DIR = os.path.expanduser("~/feature_matching_task")
IMG1_PATH = os.path.join(BASE_DIR, "data", "img1.png")
IMG2_PATH = os.path.join(BASE_DIR, "data", "img2.png")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULT_DIR, "orb_matches_ransac.png")

# =========================
# SETTINGS
# =========================
NFEATURES = 500
WARMUP_RUNS = 5
BENCH_RUNS = 50
TOP_DRAW = 100
RANSAC_THRESH = 3.0

# =========================
# LOAD IMAGES
# =========================
img1 = cv2.imread(IMG1_PATH, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(IMG2_PATH, cv2.IMREAD_GRAYSCALE)

if img1 is None:
    raise FileNotFoundError(f"img1 okunamadı: {IMG1_PATH}")
if img2 is None:
    raise FileNotFoundError(f"img2 okunamadı: {IMG2_PATH}")

# =========================
# ORB + MATCHER
# =========================
orb = cv2.ORB_create(
    nfeatures=NFEATURES,
    scaleFactor=1.2,
    nlevels=8
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

extract_times = []
match_times = []
total_times = []

last_kp1 = None
last_kp2 = None
last_matches = None
last_good_pts1 = None
last_good_pts2 = None

def compute_ransac_metrics(pts1, pts2, ransac_thresh=3.0):
    if len(pts1) < 4 or len(pts2) < 4:
        return {
            "homography_found": False,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": None,
            "mask": None,
        }

    H, mask = cv2.findHomography(
        pts1.reshape(-1, 1, 2),
        pts2.reshape(-1, 1, 2),
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

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    proj = (H @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    errors = np.linalg.norm(proj - pts2, axis=1)
    reproj_error = float(np.mean(errors[mask])) if inliers > 0 else None

    return {
        "homography_found": True,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "reproj_error": reproj_error,
        "mask": mask,
    }

# =========================
# WARM-UP
# =========================
for _ in range(WARMUP_RUNS):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None:
        _ = bf.match(des1, des2)

# =========================
# BENCHMARK
# =========================
for _ in range(BENCH_RUNS):
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    t2 = time.perf_counter()

    if des1 is None or des2 is None:
        continue

    matches = bf.match(des1, des2)
    t3 = time.perf_counter()

    extract_ms = (t2 - t1) * 1000.0
    match_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0

    extract_times.append(extract_ms)
    match_times.append(match_ms)
    total_times.append(total_ms)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    last_kp1 = kp1
    last_kp2 = kp2
    last_matches = matches
    last_good_pts1 = pts1
    last_good_pts2 = pts2

if not total_times or last_matches is None:
    raise RuntimeError("Benchmark sırasında geçerli eşleşme üretilemedi.")

# =========================
# QUALITY / RANSAC
# =========================
quality = compute_ransac_metrics(last_good_pts1, last_good_pts2, ransac_thresh=RANSAC_THRESH)

# =========================
# METRICS
# =========================
avg_extract = float(np.mean(extract_times))
avg_match = float(np.mean(match_times))
avg_total = float(np.mean(total_times))
min_total = float(np.min(total_times))
max_total = float(np.max(total_times))
hz = 1000.0 / avg_total if avg_total > 0 else 0.0

distances = [m.distance for m in last_matches]
avg_distance = float(np.mean(distances)) if distances else 0.0
min_distance = float(np.min(distances)) if distances else 0.0
max_distance = float(np.max(distances)) if distances else 0.0

confidences = [1.0 - (d / 100.0) for d in distances]
confidences = [max(0.0, min(1.0, c)) for c in confidences]
avg_confidence = float(np.mean(confidences)) if confidences else 0.0
min_confidence = float(np.min(confidences)) if confidences else 0.0
max_confidence = float(np.max(confidences)) if confidences else 0.0

# =========================
# DRAW ONLY INLIERS
# =========================
draw_matches = []
if quality["mask"] is not None:
    inlier_indices = np.where(quality["mask"])[0][:TOP_DRAW]
    for idx in inlier_indices:
        draw_matches.append(last_matches[idx])

vis = cv2.drawMatches(
    img1, last_kp1,
    img2, last_kp2,
    draw_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

os.makedirs(RESULT_DIR, exist_ok=True)
cv2.imwrite(OUT_PATH, vis)

# =========================
# PRINT
# =========================
print("=" * 60)
print("ORB + BFMatcher RESULT")
print("=" * 60)
print(f"img1 keypoint sayisi      : {len(last_kp1)}")
print(f"img2 keypoint sayisi      : {len(last_kp2)}")
print(f"toplam match sayisi       : {len(last_matches)}")
print(f"gosterilen inlier match   : {len(draw_matches)}")
print(f"min confidence            : {min_confidence:.3f}")
print(f"max confidence            : {max_confidence:.3f}")
print(f"ortalama confidence       : {avg_confidence:.3f}")
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

print("\nEXCEL SATIRI:")
print("Algoritma\tCihaz / Donanım\tToplam Süre (sn)\tÇalışma Frekansı (Hz)\tToplam Eşleşme Sayısı\tÇizilen İnlier Sayısı\tOrtalama Güven Skoru")
print(f"ORB + BFMatcher\tYerel PC\t{avg_total/1000.0:.5f}\t{hz:.2f}\t{len(last_matches)}\t{len(draw_matches)}\t{avg_confidence:.3f}")