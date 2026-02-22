import cv2
import numpy as np


# --- Helper functions ---
def uicm(img):
    r, g, b = cv2.split(img.astype("float"))
    rg = r - g
    yb = (r + g) / 2 - b
    return np.sqrt(np.var(rg) + np.var(yb))

def uism(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return np.mean(np.sqrt(gx**2 + gy**2))




def uiconm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.std(gray)

def compute_uiqm(img):
    return 0.0282 * uicm(img) + 0.2953 * uism(img) + 3.5753 * uiconm(img)