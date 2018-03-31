## <a name="top"></a> Deep Learning: Advanced Lane Finding [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

### Pipeline (video)
#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Here's a [link to my video result](./project_video.mp4)

### Discussion
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubic Points](https://review.udacity.com/#!/rubrics/476/view)
### Camera Calibration
Have the camera matrix and distortion coefficients been computed correctly and checked on the calibration test image?

The code for this step is contained in the first code cell of the Python library located in `/lib/camera_calibration.py`.
To calibrate camera I started with objpoints, imgpoints. Prepared object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). I used 9x6 chess board from `camera_cal`. For each image in camera_cal folder I use OpenCv function to convert image to grey colo cheme `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`. To find chess board corners on each image I used `ret, corners = cv2.findChessboardCorners(gray, (9,6), None)`. If `ret` was `True` conrners were found and I added them to the array of `imgpoints`. Than I draw chess board conrners on the output image with `cv2.drawChessboardCorners(image, (nx,ny), corners, ret)`. After calibrating camera with the steps discribed above I received `mtx` and `dist` values by running `mtx, dist = calibrate_camera()`. For example output images I used calibration camera results to run it on the set of images to show that it works, I used `dst = cv2.undistort(img, mtx, dist, None, mtx)`. Here is what I got:
![chess board corners](/output_images/pre_process_steps/corners_found11.jpg "Chess board corners")
![undistort transform](/output_images/pre_process_steps/undistort_calibration11.jpg "Undistort Transform")
![undistort transform road](/output_images/pre_process_steps/undistort_transform.jpg "Undistort Transform Real Road Lane Lines")
More image examples in `/output_images/pre_process_steps/` folder.
### Pipeline (single images)
For each of the images in `test_images` I created a pipaline. The frist step was to get saved calibration camera results from `output_images/calibrate_camera.p` 
```python
with open("output_images/calibrate_camera.p", "rb") as f:
        save_dict = pickle.load(f)
    mtx = save_dict["mtx"]
    dist = save_dict["dist"]
```
and apply it to the images to undistored each images with OpenCv function `undist = cv2.undistort(img_in, mtx, dist, None, mtx)`.
Here is an example ![test1 undistort](/output_images/example/test1_undistort.png)

Step two was to take undistoredted image and apply combined threshold with `img, combined2, abs_bin, mag_bin, dir_bin, hls_s_bin, lab_b_bin, hls_l_bin = combined_thresh(undist)`
Here is an axample ![combined threshold](output_images/combined_threshold.jpg "Apply Combined threshold")

As a finale image pre processgin step I did perspective transform with `binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)`, `img` is the output of thepreviouse function `combined_thresh()`.

Here is an aexample ![pipeline warped](/output_images/pipeline_warped.jpg "Perspective transform")

Next step was to draw a sliding box on found lane line to locate just the lane line and exclude the rest of the pixels that do not belond to the lane.
![pipeline sliding box](/output_images/pipeline_sliding_box.jpg "Line finding, sliding box")
`line_fit(binary_warped)` will try to find a line on the image. 
![straight_lines2_polyfit](/output_images/example/straight_lines2_polyfit.png "Polyfit cunstion on straight lanes")
Try to fit left and right lines with
```python
    left_fit = left_line.add_fit(left_fit)
    right_fit = right_line.add_fit(right_fit)
```

Calculate line curvature `left_curve, right_curve = calculate_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)`.
I used these formulars to calculate it: 

<img src="http://latex.codecogs.com/gif.latex?f(y)&space;=&space;Ax^{2}&space;&plus;&space;Bx&space;&plus;&space;C" title="f(y) = Ax^{2} + Bx + C" />

<img src="http://latex.codecogs.com/gif.latex?{f}'(y)&space;=&space;\frac{\mathrm{d}&space;x}{\mathrm{d}&space;y}&space;=&space;2Ax&space;&plus;&space;B" title="{f}'(y) = \frac{\mathrm{d} x}{\mathrm{d} y} = 2Ax + B" />

<img src="http://latex.codecogs.com/gif.latex?{f}''(y)&space;=&space;\frac{\mathrm{d}&space;x^{2}}{\mathrm{d}&space;y^{2}}&space;=&space;2A" title="{f}''(y) = \frac{\mathrm{d} x^{2}}{\mathrm{d} y^{2}} = 2A" />

<img src="http://latex.codecogs.com/gif.latex?R_{curve}&space;=&space;\frac{\left&space;|&space;(1&space;&plus;&space;(2Ay&space;&plus;&space;B)^{2})^{3/2}&space;\right&space;|}{\left&space;|&space;2A&space;\right&space;|}" title="R_{curve} = \frac{\left | (1 + (2Ay + B)^{2})^{3/2} \right |}{\left | 2A \right |}" />

As a final step try to find vechicle offset `vehicle_offset = calculate_vehicle_offset(undist, left_fit, right_fit)`

### Pipeline (video)
The image processing pipeline that was established to find the lane lines in images successfully processes the project video. The pipeline correctly maps out curved lines and does not fail when shadows or pavement color changes are present.

In the first few frames of video, the algorithm performs a search without prior assumptions about where the lines are. Once a confidence detection is achieved, that positional knowledge is used in future iterations as a starting point to find the lines.

[Project video](/output_images/video/project_video_output.mp4)
[Challenge video](/output_images/video/challenge_video_output.mp4)
[Harder Challenge video](/output_images/video/harder_challenge_video_output.mp4)

### Discussion
*  This problem involves lot of hyper-parameters that need to be tuned invididually and properly to get the correct outcome. Perhaps, use of tools like the varying hyperparameters to check the output could be beneficial.
*  During video testing, I used `subclip = VideoFileClip("project_video.mp4").subclip(41, 43)` a `subclip` to see/render only a part of the video, because rendering of the full video is very CPU heavy and time consuming.
*  The pipeline fails in low light conditions where the lanes are not visible, covered with dark shades or lane curves spontaneously and drastically. This was observed during the testing of the pipeline on Challege and Hard challege videos.
This project was based on computer vision techniques, to enhence line detection and perhaps improve algorythm performance, I would like to implement a solution to this problem using machine learning technics I acquired during previouse lessons.