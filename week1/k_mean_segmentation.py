import cv2
import numpy as np
import random
from glob import glob


def kmeans(
    X: np.ndarray,
    k: int,
    max_iters: int = 300,
    tol: float = 1e-4,
    init: str = "kmeans++",
    random_state: int | None = None
):
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # Initialize centroids
    if init == "random":
        centers = X[rng.choice(n, size=k, replace=False)]
    elif init == "kmeans++":
        centers = _kmeans_plus_plus_init(X, k, rng)
    else:
        raise ValueError("init must be 'random' or 'kmeans++'")

    for _ in range(max_iters):
        labels = _closest_center_labels(X, centers)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i)
                                else X[rng.integers(0, n)]  # handle empty cluster
                                for i in range(k)])
        shift = np.linalg.norm(centers - new_centers) / k
        centers = new_centers
        if shift < tol:
            break
    labels = _closest_center_labels(X, centers)
    inertia = ((X - centers[labels])**2).sum()
    return centers, labels, inertia


def _closest_center_labels(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    return np.argmin(dists, axis=1)


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    idx = rng.integers(0, n)
    centers[0] = X[idx]
    closest_sq_dist = np.sum((X - centers[0])**2, axis=1)
    for i in range(1, k):
        probs = closest_sq_dist / closest_sq_dist.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        new_sq_dist = np.sum((X - centers[i])**2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)
    return centers


def kmeans_segment_image_cv2(img_path: str, k: int, **kmeans_kwargs):
    img = cv2.imread(img_path)  # BGR image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, C = img_rgb.shape
    X = img_rgb.reshape(-1, C).astype(np.float32)

    centers, labels, _ = kmeans(X, k, **kmeans_kwargs)
    seg = centers[labels].reshape(H, W, C).astype(np.uint8)

    seg_bgr = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)
    return seg_bgr

if __name__ == "__main__":
    images = glob('data/BSR/BSDS500/data/images/train/*.jpg')
    img_path = random.choice(images)
    seg_img = kmeans_segment_image_cv2(img_path, k=5, init="kmeans++", random_state=0)
    cv2.imshow("Segmented Image", seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

