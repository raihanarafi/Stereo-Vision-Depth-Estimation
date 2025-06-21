import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import createRightMatcher, createDisparityWLSFilter

# --- Load stereo image pair ---
imgL = cv2.imread("left.png")   
imgR = cv2.imread("right.png")

# --- Camera Parameters ---
baseline = 0.25  # meters
hfov_deg = 75
image_width = 1024
image_height = 576
image_WIDTH = imgL.shape[1]
fx = fy = (image_WIDTH / 2) / np.tan(np.radians(hfov_deg / 2)) 

# --- Convert to grayscale for disparity calculation ---
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# --- Stereo matchers ---
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = createRightMatcher(left_matcher)
wls_filter = createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# --- Compute disparities ---
dispL = left_matcher.compute(grayL, grayR).astype(np.int16)
dispR = right_matcher.compute(grayR, grayL).astype(np.int16)
filtered_disp = wls_filter.filter(dispL, grayL, None, dispR)

# --- Disparity to depth ---
disparity = filtered_disp.astype(np.float32) / 16.0
depth_map = np.zeros_like(disparity)
depth_map[disparity > 0] = (fx * baseline) / disparity[disparity > 0]

# --- Visualization ---
disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
color_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

min_depth = 0.5 # meters
max_depth = 5.0 # meters

def onclick(event):
    if event.inaxes == axs[2]:  # Clicked on Depth Map
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth = depth_map[y, x]
            if min_depth <= depth <= max_depth:
                print(f"Clicked at ({x},{y}) → Depth: {depth:.2f} m")
                axs[2].text(x, y, f"{depth:.2f}m", color='white', fontsize=8, weight='bold')
                fig.canvas.draw()
            else:
                print(f"Clicked at ({x},{y}) → Depth {depth:.2f} m is out of valid range.")


fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
axs[0].set_title("Left Image")
axs[0].axis("off")

axs[1].imshow(color_disp)
axs[1].set_title("WLS Disparity Map")
axs[1].axis("off")

im = axs[2].imshow(depth_map, cmap='inferno')
axs[2].set_title("Depth Map (meters)")
plt.colorbar(im, ax=axs[2], label="Depth (m)")
axs[2].axis("off")

# Connect mouse click
fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()

