import os
import cv2
import numpy as np


# ---------------- UICM ---------------- #
def uicm(img):
    R = img[:, :, 2].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 0].astype(np.float32)

    rg = R - G
    yb = 0.5 * (R + G) - B

    mean_rg, std_rg = np.mean(rg), np.std(rg)
    mean_yb, std_yb = np.mean(yb), np.std(yb)

    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)


# ---------------- UISM ---------------- #
def uism(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sobelx**2 + sobely**2))


# ---------------- UIConM ---------------- #
def uiconm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


# ---------------- UIQM ---------------- #
def compute_uiqm(img):
    raw_uiqm = (
    0.0282 * uicm(img) +
    0.2953 * uism(img) +
    3.5753 * uiconm(img)
)
    return raw_uiqm / 60.0



# ---------------- Evaluation ---------------- #
def evaluate_uiqm(image_dir):
    scores = []

    files = os.listdir(image_dir)
    for f in files:
        img = cv2.imread(os.path.join(image_dir, f))
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        scores.append(compute_uiqm(img))

    print("📊 UIQM Results")
    print(f"Average UIQM : {np.mean(scores):.4f}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DIR = os.path.join(BASE_DIR, "results", "images")
    evaluate_uiqm(IMAGE_DIR)
