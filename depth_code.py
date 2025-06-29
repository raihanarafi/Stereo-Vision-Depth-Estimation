import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from cv2.ximgproc import createRightMatcher, createDisparityWLSFilter

# Load stereo images
imgL = cv2.imread("/home/raihana/Intern/stereo_depth_task/mymodel/Images/1 l.png")
imgR = cv2.imread("/home/raihana/Intern/stereo_depth_task/mymodel/Images/1 r.png")

# Stereo parameters
baseline = 0.25  # meters
hfov_deg = 75
image_WIDTH = imgL.shape[1]
fx = fy = (image_WIDTH / 2) / np.tan(np.radians(hfov_deg / 2))

# Grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# StereoSGBM + WLS Filter setup
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = createRightMatcher(left_matcher)
wls_filter = createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(10000)
wls_filter.setSigmaColor(1.0)

# Compute disparity and depth
dispL = left_matcher.compute(grayL, grayR).astype(np.int16)
dispR = right_matcher.compute(grayR, grayL).astype(np.int16)
filtered_disp = wls_filter.filter(dispL, grayL, None, dispR)

disparity = filtered_disp.astype(np.float32) / 16.0
depth_map = np.zeros_like(disparity)
depth_map[disparity > 0] = (fx * baseline) / disparity[disparity > 0]

min_depth = 0.5
max_depth = 5.0

# Create a realistic depth visualization using COLORMAP_JET
clipped_depth = np.clip(depth_map, min_depth, max_depth)
smoothed = cv2.bilateralFilter(clipped_depth.astype(np.float32), 9, 75, 75)
normalized_depth = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(normalized_depth)
color_depth = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
alpha = 0.6
blended = cv2.addWeighted(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB), 1 - alpha, color_depth, alpha, 0)


# Load YOLOv8 model
model_path = "/home/raihana/Downloads/best11_model_100l.pt"
model = YOLO(model_path)

# Run inference on the left image
results = model(imgL, conf=0.05)[0]

# Annotate detections with depth

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
        depth = depth_map[cy, cx]

        if min_depth <= depth <= max_depth:
            label = f"{depth:.2f}m"
            color = (0, 255, 0)
            label_pos = (x1, y1 - 10)
        else:
            label = "Out of range"
            color = (0, 0, 255)
            label_pos = (x1, y2 + 20)

        cv2.rectangle(imgL, (x1, y1), (x2, y2), color, 1)
        cv2.putText(imgL, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, lineType=cv2.LINE_AA)


# Matplotlib interactive click
def onclick(event):
    if event.inaxes == axs[1]:  
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth = depth_map[y, x]
            if min_depth <= depth <= max_depth:
                print(f"Clicked at ({x},{y}) → Depth: {depth:.2f} m")
                
                # Draw a white dot
                axs[1].plot(x, y, 'wo', markersize=4)

                # Write depth label near the dot
                axs[1].text(x + 5, y - 5, f"{depth:.2f}m", color='white', fontsize=8, weight='bold')

                fig.canvas.draw()
            else:
                print(f"Clicked at ({x},{y}) → Depth {depth:.2f} m is out of valid range.")

                
def onhover(event):
    if event.inaxes == axs[1] and event.xdata and event.ydata:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth = depth_map[y, x]
            depth_text.set_text(f"Depth at ({x},{y}) → {depth:.2f} m")
            fig.canvas.draw_idle()


# Display images
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
axs[0].set_title("YOLO Detection with Depth")
axs[0].axis("off")

im = axs[1].imshow(blended)
axs[1].set_title("Depth Map ")
plt.colorbar(im, ax=axs[1], label="Depth Scale")
axs[1].axis("off")

depth_text = axs[1].text(0.5, -0.12, "", transform=axs[1].transAxes,
                         ha="center", fontsize=10, color="black")

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onhover)
plt.tight_layout()
plt.show()
