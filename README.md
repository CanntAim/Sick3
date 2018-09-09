# Sick3 | [Project Home](https://cantaim.live/projects/sick3)
---
![Freestyle Clip](https://github.com/CanntAim/Sick3/blob/switch-to-optical-flow/documentation/test_freestyle_normal.gif?raw=true)

## Installation

Dependencies:

* [GRT](https://github.com/nickgillian/grt)
* [OpenCV](https://github.com/opencv/opencv)
* [OpenCV Contrib](https://github.com/opencv/opencv_contrib)

Building from Cmake:

_Excerpt from [Mirki Kiefer's blog](https://mirkokiefer.com/cmake-by-example-f95eb47d45b1?gi=6feac9901e8c)_

CMake supports out-of-source builds — so all our compiled code goes into a directory separate to the sources.

To start a build we create a new folder:

```bash
mkdir _build
cd _build
```

And call cmake with the path to the project’s root (in this case the parent folder):

```bash
cmake ..
```

This will generate build scripts using the default generator — on Linux/OSX this should be Makefiles.

By default cmake will install our build into the system directories.
 To define a custom install directory we simply pass it to cmake:

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=../_install
```

To run the build script you can simply use the Makefile:

```bash
make
make install
```

We can now run our binary from the install directory:

```bash
../_install/bin/sick3core
```

## Summary

Sick3 is Football Freestyle Combo Tracker proof-concept that aims to track basic trick combos that are composed of around the worlds, hop the worlds, and half around the worlds. We are not yet looking at more advanced No Touch combos or No Touch (multi-revolution) tricks for this first iteration. For examples as to what a basic freestyle combo may look like, refer to this excellent [video](https://www.youtube.com/watch?v=2Cb8T9QvvN4) by _YourHowToDo_ YouTube channel that helps demonstrate the idea. Definition of a standard combo is _two or tricks done consecutively with no extra touch in-between_. More advanced combos exist that essentially _skip a touch_, these are known as NT combos, but they are out-of-scope for this project.

## Technical Goal

This project seeks to track fairly advanced body gestures without the use of stereo depth perceptive cameras that are more commonly used by gesture recognition tech like the Xbox Kinect and so on. Everything here works using regular camera footage. This is of high value since most consumer level phone cameras lack the duel cameras necessary to detect depth.  

## Process Outline

The following section will outline the general process we go through to identify a sequence of consecutive tricks. This project is still in progress so the details given here are subject to change. We give a brief summary of data pre-process module

### Modules

**Tracker** - Takes footage as input and outputs optical flow segments that capture motion between two touches.

**Classifier** - Takes sequence of optical flow segments and classifies each as either being, an around the world, hop the world, half around the world, or kick-up.

### Tracker

The tracker can be in multiple states. The program states are defined below.

**Initial State** - We have not yet identified the ball or began tracking.

**Tracking** - The ball has be identified and we have started tracking it.

**Dribbling** - The ball which is being tracked is now being dribbled.

In the _**Initial State**_ the program grabs frames from the video and does some preliminary image pre-processing.

* _RGB to Gray Conversion_
* _Gaussian Blur_

We now apply background subtraction which produces a foreground mask. We use the Gaussian Mixture-based Background/Foreground Segmentation Algorithm as described in two papers by _Z.Zivkovic_, published in 2006."_Improved adaptive Gaussian Mixture Model for Background Subtraction_" and in 2004 "_Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction_" One important feature of this algorithm is that it selects the appropriate number of Gaussian distribution for each pixel. It provides better adaptability to varying scenes due illumination changes, etc. We set the learning rate to zero to maintain a static background through out the execution of the program. This generally will work fine since the execution of our program is very short so we don't need to adjust for changing light conditions. Although keep in mind that it's possible that a sudden drastic change will be disruptive.

###### Foreground Mask
![Mask](https://raw.githubusercontent.com/CanntAim/Sick3/switch-to-optical-flow/documentation/mask.png)

_Note: There is an explicit requirement for us to have foreground objects out of frame when the first frame comes in._

After obtaining the mask we apply a standard image cleaning technique. we first erode the image to delete all minor noise and then dilate. Additionally we smooth out the mask by applying a median blur. We now try and find the person in the frame by examining the mask. Our method **findPerson** discovers contours that exist in the mask and takes the largest one by area. This is because we assume the subject of the video is alone and in the fore most foreground. Due to perspective they will be largest object in the frame.

Based on whether the user of the application selects manual or automatic ball selection the next steps may or may not matter. Selecting the ball manually or auto-selection will transition the program state into _**Tracking**_. Manual selection is self explanatory. For automatic selection the process is more involved but still relatively simple. To detect the ball we exploit the fact that the ball has a circular shape and apply a feature extraction technique called [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform). We apply the feature search to the lower fifth of the bound box of the person. We select all circular shapes that might be the ball and over the course of several frames check that the feature is still present. The feature that exists sufficiently long first is the one we select.

###### Ball Feature Extraction
![Found](https://raw.githubusercontent.com/CanntAim/Sick3/switch-to-optical-flow/documentation/found_ball.png)

In the _**Tracking**_ state we set our tracker on the ball we either selected manually or automatically. OpenCV as of version 3.0 has several trackers built into the contribution library. Tracking algorithms are based on two primary models, **motion** and **appearance**. Motion models are based on maintaining object identity over continuous frames using location and velocity information on the object that is being tracked, basically predicting where the object will be next. Appearance models use knowledge of how the object looks to persist the identify of an object. This model extends on the motion model by using location, speed, and direction data to search the neighborhood of where the object is predict to be for something that looks similar to the object. Tracking algorithms unlike detection or recognition algorithms are on line, they are trained in real-time during run time.

_The built-in algorithms:_

* _BOOSTING Tracker_
* _MIL Tracker_
* _KCF Tracker_
* _TLD Tracker_
* _MEDIANFLOW Tracker_
* _GOTURN Tracker_

Out of the built in algorithms we are currently using _TLD_ (which stands for tracking, learning, and detection). From the author’s paper:

>“_The tracker follows the object from frame to frame. The detector localizes all appearances that have been observed so far and corrects the tracker if necessary. The learning estimates detector’s errors and updates it to avoid these errors in the future._”

The algorithm does have some stability issues as it jumps around quite a bit (we fix this with smoothing). The flip side of this is that algorithm is able to track large motions and long term occlusion.

Once we started tracking the ball we have to wait for the user to enter the _**Dribbling**_ state. The dribbling state is defined as the ball entering an oscillating vertical y-position trend in relation to time. The second order derivative of this metric is the velocity. We trigger the entry into this state by having the ball go past a certain vertical position (around knee height). Because generally human proportions are consistent this is a safe threshold to set using height information of the person which we get when finding the person.

##### Smoothing and Segmentation

As mentioned previously, The _TLD Tracker_ has a tendency to jump around a lot. This leads to position and velocity metrics that are less than reliable. To fix this we deploy a custom [smoothing](https://en.wikipedia.org/wiki/Smoothing) method.

To smooth we first need to populate a size _k_ window(s) with raw position, velocity, and acceleration information. To get velocity from position we simply take difference between the current position and the previous position. The same is repeated to get acceleration but using velocity. A caveat is that we need at _least_ two items in our position window before we can calculate the first velocity item. This means that the velocity is calculated on delay of one frame and acceleration on a delay of two frames. The window operates as a queue that pushes new frames in and pops frames out when the queue fills up. This means that there is _start-up time_.

Alongside the raw data windows we have a smoothing window to which we pass a discrete weight function and the raw data. The size of the weight function will be adjusted to be size of window, we do simple linear interpolation to find float point values between the weights that are provided. The smoothing function is produced on heavy delay since we need to wait for the queue start-up phase to finish at which point we are calculating velocity value for the frame that will typically be _(Window Size)/2_ frames in the past from the current frame. We can pick any kind of weight distribution, even uniform, although realistically we will want something closer to normal.

Another simple trick we use to avoid irregular results is throwing away data points that deviate to far from the prior trajectory. If the position of the ball in the current frame is unrealistically far from where it was prior we don't record the instantaneous velocity value that got it to that outlying position. Off course the tracker may stay in the incorrect position for multiple frames, Because we are looking at velocity, and the actual change in position between frames at an fps of 30 to 120 is small, these values while incorrect are still generally closer to what they should be and thus don't throw off the averaging.

Using our smoothed curve we segment our video based on when the balls velocity flips from the positive to negative direction. This happens whenever the ball is kicked back up.

##### Capturing Optical Flow Map

OpenCV has a built in module for doing optical flow using various methods. We use the Gunnar Farneback’s Method. While there isn't a good explanation of that particular method in the documentation there is a decent explanation of optical flow using [Lucas Kanade Method](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html).
Optical flow operates on the assumption that A) The pixel intensities of an object do not change between consecutive frames. B) Neighboring pixels have similar motion.

We draw the flow map for the entire area of the image, currently, this will be changed to only draw the map on the part that is relevant. That part is pixels that within the area of foreground mask as calculated by our earlier background subtraction.

We want to track flow for the duration of time between two touches. Because of this we want to add the difference of current image to the **summation** of all images up to now. This additive process reset after every touch. We also want to account for the temporal component to motion by coloring the map differently along some generic gradient for each frame. We use the HSV scale where each frame we increment the Hue by some amount. The result is shown below:

###### Original Segmented Video
![htw](https://github.com/CanntAim/Sick3/blob/switch-to-optical-flow/documentation/htw.gif?raw=true)

###### Resulting Flow Map
![flowmap](https://github.com/CanntAim/Sick3/blob/switch-to-optical-flow/documentation/htw_flowmap.jpg?raw=true)

Please keep in mind that the work here is in progress. The results of our tracker are currently not ideal and there will be on going work to improve it. This is at best in pre-alpha phase.

### Classifier

TODO
