#ifndef SICK3CORE_H
#include <deque>
#include <list>
#include <stdio.h>
#include <GRT/GRT.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace GRT;
using namespace std;
using namespace cv;

void clean(Mat &mask);

/* BACKGROUND SUBTRACTION*/
/*
/* pMOG2 - pointer to MOG2 module

Initializes MOG2 and sets parameters for it.
*/
void backgroundSubtraction(Ptr<BackgroundSubtractorMOG2> &pMOG2);

/* DRAW BALL */
/*
/* frame - which frame to draw on
/* ballRectangle - rectangle to draw

Draws a bounding box around the ball.
*/
void drawBall(Mat &frame, Rect2d ballRectangle);

/* DRAW PERSON */
/*
/* frame - which frame to draw on
/* personRectangle - rectangle to draw

Draws a bounding box around the person.
*/
void drawPerson(Mat &frame, Rect personRectangle);

/* POPULATE WINDOW */
/*
/* newBox - current frame's ball bounding rectangle
/* oldBox - previous frame's ball bounding rectangle
/* smooth - window for smoothed data
/* acceleration - window for ball acceleration data
/* velocity - window for ball velocity data
/* position - window for ball position data
/* maxBandwidth - the max size for windows that will be populated
/* toSmooth - code (1,2,3) to decide which window to use to create smooth
/* window.

We monitor the position of the ball as it's moving up and down. Sudden, drastic, changes in position
we ignore. If the center of the new box doesn't fall into the box in the previous frame it's unlikely to
be right. We continue logging in the window the position right afterward until the next significant change.
We don't know when the tracker gets back to tracking the ball again since two concurrent drastic changes doesn't
tell us if the tracker corrected itself or if it jumped to another incorrect location.

We create a window first for position and then using position create a window for velocity and then acceleration. The velocity
window elimanates large outliers using the method above. Our velocity window is still noisy. The tracker, while sufficently
accurate, still tends to be erratic. While the movement of the ball tends to a general direction it's constantly making
small adjustments counter to it's overall movement. For this we apply smoothing. Meaning we take the weighted
average of the window (using an applied kernel). Using these "smooth" value we can monitor a change in direction of velocity.
The means monitoring when velocity goes from + to - or from - to +. You can choose to smooth position or velocity
*/
void populateWindow(Rect2d newBox, Rect2d oldBox,
  deque<float> &smooth,
  deque<int> &acceleration, deque<int> &velocity, deque<int> &position,
  vector<float> weights, int toSmooth);

/* CALCULATE DIFFERENCE */
/*
/* cur - current value
/* prev - previous value

Use to calculate the difference between two values. Is generic because we might
calculate difference between two positions to get velocity or the difference between
two velocities to get acceleration.
*/
int calculateDifference(int cur, int prev);

/* CALCULATE WINDOW */
/*
/* window - the window which we want to calculate for.

We can calculate for any window this can be position, velocity, acceleration, smooth,
etc.
*/
float calculateWindow(deque<int> window, vector<float> weights);

/* CALCULATE BALLBOUND */
/*
/* ball - The Vec3f represnting a ball.

Get the bound around the discovered ball.
*/
Rect2d ballBound(Vec3f &ball);

/* CALCULATE FINDPERSON */
/*
/* mask - The mask created by the MOG2 algorithm.

Get the bound around the person in the MOG2 mask.
*/
Rect findPerson(Mat &mask);

/* CALCULATE FINDBALL */
/*
/* grey - The greyscale frame,
/* personRectangle - rectangle around found person
/* potentialBalls - container for the potential balls that we found.

Get the bound around the discovered ball.
*/
tuple<Point,Vec3f,int> findBall(Mat &grey,
  Rect personRectangle,
  vector<tuple<Point,Vec3f,int>> &potentialBalls);

  /* CHECK DIRECTION CHANGE */
  /*
  /* buffer - A buffer for data.

  Check if direction of the ball has changed.
  */
  bool checkDirectionChange(deque<float> &buffer);

  /* CHECK THAT USER BEGAN DRIBBLING */
  /*
  /* flag - a boolean flag to check.
  /* verticalPostion - the vertical psotion of the ball

  Check if direction of the ball has changed.
  */
  bool checkDribbling(bool &flag, int verticalPostion, Rect personRectangle);
#endif
