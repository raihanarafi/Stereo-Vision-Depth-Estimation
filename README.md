Stereo Vision Depth Estimation with YOLOv8 Object Detection

This project performs depth estimation from a stereo image pair using OpenCV's StereoSGBM algorithm with WLS filtering, and overlays detected rocks using a custom-trained YOLOv8 model.

Features:
Accurate depth estimation in the range 0.5m to 5m

YOLOv8-based rock detection

Realistic, colorful depth visualization

Interactive clicks to display depth at any pixel

Depth map blended with the original image

Displays:
Left image

Depth map (with color bar)

Interactive click to show real-world depth at any point on the disparity map

Tools & Libraries Used:
Python

OpenCV (with contrib) – for stereo matching and visualization

NumPy – for fast matrix operations

Matplotlib – for displaying images and handling interaction

YOLOv8 (Ultralytics) – for object detection

Camera Setup:
Baseline: 25 cm

HFOV: 75°

Image resolution: 1024x576

Focal length is calculated from HFOV and image width.

How It Works:
Load stereo image pair (e.g., left.png and right.png)

Convert images to grayscale

Compute disparity map using StereoSGBM

Apply WLS filter to smooth and refine the disparity

Convert disparity to real-world depth using:
Depth = fx * Baseline / Disparity

Colorize the depth map using COLORMAP_JET

Blend the depth map with the original left image

Run YOLOv8 on the left image to detect rocks

Display detection + interactive depth info

Install dependencies:

    pip install opencv-contrib-python matplotlib numpy

Run the script and interact with the depth map

 Notes

    You must calibrate your stereo cameras for best accuracy (intrinsics + extrinsics).

    YOLOv8 model must be custom-trained for rock detection and saved as a .pt file.

    Works best when stereo images are well-aligned and rectified.
