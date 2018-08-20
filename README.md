## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./writeUpImages/01_chessboard_calibration.png
[image2]: ./writeUpImages/02_chessboard_PT.png
[image3]: ./writeUpImages/03_road_undist.png
[image4]: ./writeUpImages/04_road_PT.png
[image5]: ./writeUpImages/05_sobel_threshold_X.png
[image6]: ./writeUpImages/06_mag_threshold.png
[image7]: ./writeUpImages/07_dir_threshold.png
[image8]: ./writeUpImages/08_solbel_combined.png
[image9]: ./writeUpImages/09_yellow_line_PT.png
[image10]: ./writeUpImages/10_yellow_line_L.png
[image11]: ./writeUpImages/11_yellow_line_S.png
[image12]: ./writeUpImages/12_yellow_line_BlueYellow.png
[image13]: ./writeUpImages/13_yellow_line_color_combined.png
[image14]: ./writeUpImages/14_hist.png
[image15]: ./writeUpImages/15_sliding_window.png
[image16]: ./writeUpImages/16_prev.png
[image17]: ./writeUpImages/17_current.png
[image18]: ./writeUpImages/18_sliding_window_prev.png
[image19]: ./writeUpImages/19_lane_overlay.png
[image20]: ./writeUpImages/20_data_overlay.png

[video1]: ./project_video.mp4 "Video"
[video1]: ./project_video.mp4 "Video"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Modified from the sample code given from "./examples/example.ipynb", I start by preparing "object points", 
which will be the (x, y, z) coordinates of the chessboard corners in the reality. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, 
such that the object points are the same for each calibration image.  Thus, 

`objp` is just a replicated array of coordinates, 

`objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Furthermore, I applied a perspective transform on the chessboard image to get this:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

#### For detailed steps and how each functions were made, please refer to my [.ipynb](https://github.com/MaxwellFX/Advanced_Lane_Finding/blob/master/Advanced_Lane_Finding.ipynb) file

Here is the processing steps I have taken to create a binary thresholded image:

* Perform a perspective transform to 'straighten' the image:

![alt text][image9]

* Then, I extracted the L channel information and applied sobel magnitude thresholding on it

![alt text][image10]

* And, I extracted the S_channel and BlueYellow_channel and applied color threshold on them respectively:

![alt text][image11]
![alt text][image12]

* Lastly, I combined them into a color binary image:

![alt text][image12]

* Here is the code that does the combination:

```python
def color_thresh_pipeline(unwarped_img, sx_thresh=(20, 185), S_thresh=(125, 255), B_thresh = (220, 255)):
    img = np.copy(unwarped_img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    B_channel = get_blueYellow(img)
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = color_thresh(s_channel, S_thresh)
    
    # Stack each channel to view each individual contribution
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_corlor_binary = np.zeros_like(sxbinary)
    combined_corlor_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_corlor_binary, color_binary
```

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# Define the transformation matrix parameters
rows, cols = undistorted_roadImg.shape[:2]

left_ref_x = (cols * 2)//7
right_ref_x = (cols * 5)//7

src = np.float32([(588, 454),
                  (695, 454), 
                  (1000, 650), 
                  (313, 650)])

dst = np.float32([(left_ref_x,0),
                  (right_ref_x, 0),
                  (right_ref_x, rows),
                  (left_ref_x, rows)])

# Perform the perspective transformation for road image
unwarped_roadImage, _, _ = perspective_transform(undistorted_roadImg, src, dst)

# Plot comparison
image_comparison_plot(undistorted_roadImg, 'Undistorted Road Image', 
                      unwarped_roadImage, 'Perspective Transformed', 
                      oScopeSetting = {'src': src, 'dst': dst})
```

![alt text][image4]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 588, 454      | 365, 0        | 
| 695, 454      | 914, 0      |
| 1000, 650     | 914, 720      |
| 313, 650      | 365, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Following the tutorial:

```python
def plot_ploy_fit(unwarped_binary):
    out_img, x_fits, y_plot = fit_polynomial(unwarped_binary)
    plt.figure(figsize=(40,10))
    plt.plot(x_fits[0], y_plot, color='yellow')
    plt.plot(x_fits[1], y_plot, color='yellow')
    plt.imshow(out_img)
    plt.show()
```

where `fit_polynomial` and its associated functions were straight off from the tutorial

And here is the resulting image for poly fit:

![alt text][image15]

I used these two similar images as current and previous frame:

![alt text][image16]
![alt text][image17]

And here is the image for using previous polyfit to skip the sliding window:

![alt text][image18]

Again, for detailed steps, please refer to the [.ipynb](https://github.com/MaxwellFX/Advanced_Lane_Finding/blob/master/Advanced_Lane_Finding.ipynb) file

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```python
def generate_data(combined_color_binary, 
                  ym_per_pix, xm_per_pix, 
                  l_lane_indices, r_lane_indices):
    # cache value in case if left or right lanes were not found through thresholding
    global temp_left_cr
    global temp_right_cr
    
    rows, _ = combined_color_binary.shape
    
    ploty = np.linspace(0, rows - 1, num= rows)# to cover same y-range as image
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined_color_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    leftx = nonzerox[l_lane_indices]
    lefty = nonzeroy[l_lane_indices] 
    rightx = nonzerox[r_lane_indices]
    righty = nonzeroy[r_lane_indices]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        temp_left_cr = left_fit_cr
        temp_right_cr = right_fit_cr
    else:
        # using cached value
        left_fit_cr = temp_left_cr
        right_fit_cr = temp_right_cr
    
    return ploty, left_fit_cr, right_fit_cr

def measure_curvature_real(combined_color_binary, 
                          leftFit, rightFit, 
                          left_lane_indices, right_lane_indices,
                          ym_per_img = 13, xm_per_img = 3.7):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    left_curverad = 0
    right_curverad = 0 
    center_dist = 0
    
    rows, cols  = combined_color_binary.shape
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = ym_per_img / rows 
    xm_per_pix = xm_per_img / 548 # This is the number specific was set in accordance to my image transformation setting
    
    ploty, left_fit_cr, right_fit_cr = generate_data(combined_color_binary, 
                                                  ym_per_pix, xm_per_pix, 
                                                  left_lane_indices, right_lane_indices)
    
    # Choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculate the R_curve
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Distance from center is image x midpoint - mean of left_fit_x_int and right_fit_x_int intercepts 
    if leftFit is not None and rightFit is not None:
        car_position = combined_color_binary.shape[1]/2
        left_fit_x_int = leftFit[0] * rows**2 + leftFit[1] * rows + leftFit[2]
        right_fit_x_int = rightFit[0] * rows**2 + rightFit[1] * rows + rightFit[2]
        lane_center_position = (left_fit_x_int + right_fit_x_int) /2
		# calculated the center offset
        center_offset = (car_position - lane_center_position) * xm_per_pix
        
    return left_curverad, right_curverad, center_offset

```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

For detailed implementation, please refer to the [.ipynb](https://github.com/MaxwellFX/Advanced_Lane_Finding/blob/master/Advanced_Lane_Finding.ipynb) file

![alt text][image20]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

Here's a [not so well performed challenged video](./challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My biggest mistake so far was not write the code in object oriented style. Especially in the later stage, I found it was much harder to tune a specific parameters just none of my previous functions and data were object oriented. 

To improve on the project, a better scope/masking algorithm is need. The car needs to learn what region of interests to search based on the given image. Such algorithm was not implemented in project, thus it was very difficult to find the correct lane information, especially for those challenging videos. With a more advanced scope searching algorithm, even for current lane search implementation should do a superb job. But expect that kind of task to be reserved in term 2/3 of the self-driving car nano degree.
