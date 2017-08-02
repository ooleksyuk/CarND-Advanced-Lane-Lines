## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![Straight Lines Annotated Image](/output_images/examples/straight_lines1_annotated.png "Straight Lines Annotated Image")

## [Rubic Points](https://review.udacity.com/#!/rubrics/476/view)
### Camera Calibration
Have the camera matrix and distortion coefficients been computed correctly and checked on the calibration test image?

The code for this step is contained in the first code cell of the Python library located in `/lib/camera_calibration.py`.
To calibrate camera I started with objpoints, imgpoints. Prepared object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). I used 9x6 chess board from `camera_cal`. For each image in camera_cal folder I use OpenCv function to convert image to grey colo cheme `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`. To find chess board corners on each image I used `ret, corners = cv2.findChessboardCorners(gray, (9,6), None)`. If `ret` was `True` conrners were found and I added them to the array of `imgpoints`. Than I draw chess board conrners on the output image with `cv2.drawChessboardCorners(image, (nx,ny), corners, ret)`. After calibrating camera with the steps discribed above I received `mtx` and `dist` values by running `mtx, dist = calibrate_camera()`. For example output images I used calibration camera results to run it on the set of images to show that it works, I used `dst = cv2.undistort(img, mtx, dist, None, mtx)`. Here is what I got:
![chess board corners](/output_images/pre_process_steps/corners_found11.jpg "Chess board corners")
![undistort transform](/output_images/pre_process_steps/undistort_calibration11.jpg "Undistort Transform")
![undistort transform road](/output_images/pre_process_steps/undistort_transform.jpg "Undistort Transform Real Road Lane Lines")
More image examples in `More image examples in `.
