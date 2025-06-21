Stereo Vision Depth Estimation with OpenCV

This project performs depth estimation using a pair of stereo images.
It uses OpenCV's StereoSGBM algorithm with WLS filtering to generate a smooth disparity map and calculate depth values from it.

Features:
Depth estimation range: 0.5m to 5m

Works with real stereo images captured from IP webcam or phone

WLS filtering for enhanced disparity quality

Displays:
Left image

Colored disparity map

Depth map (with color bar)

Interactive click to show real-world depth at any point on the disparity map

Tools & Libraries Used:
Python

OpenCV (opencv-contrib-python)

NumPy

Matplotlib

Camera Setup:
Baseline: 25 cm

HFOV: 75°

Image resolution: 1024x576

Focal length is calculated from HFOV and image width.

How It Works:
Load stereo image pair (left.png, right.png)

Compute disparity using StereoSGBM

Apply WLS filtering to refine disparity

Convert disparity to real-world depth using the pinhole camera model:
    Depth=fx⋅Baseline / Disparity
    
Visualize using Matplotlib

Click on the image to see depth value printed and annotated

How to Use:
Place stereo images as left.png and right.png

Install dependencies:

pip install opencv-contrib-python matplotlib numpy

Run the script and interact with the depth map
