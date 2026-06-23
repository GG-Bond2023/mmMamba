import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. Set core parameters (added angle range) --------------------------
# Camera image parameters
CAM_H = 480  # Height
CAM_W = 640  # Width
CAM_CX = CAM_W // 2  # Polar origin (image center) x
CAM_CY = CAM_H // 2  # Polar origin (image center) y
MAX_R = np.sqrt(CAM_CX ** 2 + CAM_CY ** 2)  # Maximum radius of camera image

# Polar segmentation map parameters
POLAR_R = 240  # Number of radial samples
POLAR_THETA = 360  # Number of angular samples (0~360°)
# Key: polar angle range of ±45° in front of the camera (radians)
FRONT_THETA_MIN = np.pi / 4  # 45°
FRONT_THETA_MAX = 3 * np.pi / 4  # 135°
# Semantic classes (only displayed within ±45°)
SEMANTIC_CLASSES = {
    0: (0, 0, 0),  # Background (black)
    1: (0, 0, 255),  # Front semantic (red, within ±45°)
}

# -------------------------- 2. Simulate polar semantic segmentation map (semantic only within ±45°) --------------------------
polar_semantic = np.zeros((POLAR_R, POLAR_THETA, 3), dtype=np.uint8)

for theta_col in range(POLAR_THETA):
    # Column index → polar angle θ (0~2π radians)
    theta_rad = (theta_col / POLAR_THETA) * 2 * np.pi

    # Assign semantics only within the front ±45° range (π/4 ~ 3π/4)
    if FRONT_THETA_MIN <= theta_rad <= FRONT_THETA_MAX:
        polar_semantic[:, theta_col, :] = SEMANTIC_CLASSES[1]  # Red semantic
    # Other angles remain background (black)


# -------------------------- 3. Conversion: Keep only front ±45° semantics --------------------------
def bilinear_interpolate(img, x, y):
    """Bilinear interpolation (reusing the previous function)"""
    h, w = img.shape[:2]
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return SEMANTIC_CLASSES[0]

    dx = x - x0
    dy = y - y0
    val = (1 - dx) * (1 - dy) * img[y0, x0] + \
          dx * (1 - dy) * img[y0, x1] + \
          (1 - dx) * dy * img[y1, x0] + \
          dx * dy * img[y1, x1]
    return val.astype(np.uint8)


# Vectorized computation (replacing for loop for efficiency)
x, y = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))  # Generate all pixel coordinates of the camera image
x_centered = x - CAM_CX  # Centered coordinate x (positive to the right)
y_centered = CAM_CY - y  # Centered coordinate y (positive upward, reversing original y-axis)

# Compute polar coordinates (r, θ) for each pixel
r = np.sqrt(x_centered ** 2 + y_centered ** 2)  # Radius
theta = np.arctan2(y_centered, x_centered)  # Polar angle (-π ~ π) → no need to add π, directly compare range

# Initialize camera semantic image (all background)
cam_semantic = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

# Key: filter pixels within "front ±45° and radius within valid range"
valid_mask = (theta >= FRONT_THETA_MIN) & (theta <= FRONT_THETA_MAX) & (r <= MAX_R)

# Compute polar indices and interpolate only for valid pixels
if np.any(valid_mask):
    # Extract valid pixel coordinates
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_xc = x_centered[valid_mask]
    valid_yc = y_centered[valid_mask]
    valid_r = r[valid_mask]
    valid_theta = theta[valid_mask]

    # Polar (r, θ) → polar map indices (float)
    r_idx = (valid_r / MAX_R) * (POLAR_R - 1)  # Map radius to polar map row
    theta_idx = (valid_theta / (2 * np.pi)) * (POLAR_THETA - 1)  # Map angle to polar map column

    # Interpolate and sample for each valid pixel
    for i in range(len(valid_x)):
        y_cam = valid_y[i]
        x_cam = valid_x[i]
        cam_semantic[y_cam, x_cam, :] = bilinear_interpolate(
            polar_semantic, theta_idx[i], r_idx[i]
        )

# -------------------------- 4. Visualize results (highlight ±45° range) --------------------------
plt.figure(figsize=(12, 6))

# Subplot 1: Polar semantic segmentation map (only middle ±45° in red)
plt.subplot(1, 2, 1)
plt.title("Polar Semantic Segmentation Map\n(only front ±45° has semantics, red=valid class, black=background)")
plt.imshow(cv2.cvtColor(polar_semantic, cv2.COLOR_BGR2RGB))
plt.xlabel("Angle (0~360°, red segment=45°~135°)")
plt.ylabel("Radius (0~MAX_R)")

# Subplot 2: Converted camera semantic image (only top center ±45° in red)
plt.subplot(1, 2, 2)
plt.title("Camera Semantic Image\n(only front ±45° shows semantics, red=valid class, black=background)")
plt.imshow(cv2.cvtColor(cam_semantic, cv2.COLOR_BGR2RGB))
# Draw guiding lines: mark front ±45° boundaries (extending outward from image center)
plt.axline((CAM_CX, CAM_CY), slope=np.tan(np.pi - FRONT_THETA_MIN), color='white', linestyle='--',
           alpha=0.8)  # Left boundary (135°)
plt.axline((CAM_CX, CAM_CY), slope=np.tan(np.pi - FRONT_THETA_MAX), color='white', linestyle='--',
           alpha=0.8)  # Right boundary (45°)
plt.xlabel("Camera X-axis (right)")
plt.ylabel("Camera Y-axis (down)")

plt.tight_layout()
plt.show()