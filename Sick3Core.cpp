#include <queue>
#include <list>
#include <stdio.h>
#include <GRT/GRT.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace GRT;
using namespace std;
using namespace cv;

void clean(Mat &mask){
  erode(mask,mask,Mat(),Point(-1, -1),2,1,1);
  dilate(mask,mask,Mat(),Point(-1, -1),5,1,1);
  medianBlur(mask,mask,21);
}

void backgroundSubtraction(Ptr<BackgroundSubtractorMOG2> &pMOG2){
  pMOG2 = createBackgroundSubtractorMOG2();
  pMOG2->setNMixtures(1);
  pMOG2->setVarThreshold(10.0);
  pMOG2->setDetectShadows(false);
}

Rect findPerson(Mat &mask){
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  Rect maxPolyRectangle;
  findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
  int maxPolyRectangleArea = 0;
  vector<Rect> bounds(contours.size());
  vector<vector<Point>> contoursPoly(contours.size());
  for(int i = 0; i < contours.size(); i++){
    approxPolyDP(cv::Mat(contours[i]), contoursPoly[i], 3, true);
    for(int i = 0; i < contoursPoly.size(); i++){
      if(boundingRect(Mat(contoursPoly[i])).area() > maxPolyRectangleArea){
        maxPolyRectangle = boundingRect(Mat(contoursPoly[i]));
        maxPolyRectangleArea = maxPolyRectangle.area();
      }
    }
  }
  return maxPolyRectangle;
}

tuple<Point,Vec3f,int> findBall(Mat &grey, Rect maxPolyRectangle, vector<tuple<Point,Vec3f,int>> &potentialBalls){
  vector<Vec3f> circles;
  HoughCircles(Mat(grey,maxPolyRectangle), circles, CV_HOUGH_GRADIENT, 1, grey.rows/8,200,25,0,100);

  for( size_t x = 0; x < circles.size(); x++ ){
    double heightThreshold = (double)(maxPolyRectangle.height)/5.0;
    if (circles[x][1] > maxPolyRectangle.height - heightThreshold){
      bool isNewPotentialBall = true;
      Point center(cvRound(circles[x][0]), cvRound(circles[x][1]));
      for(size_t y = 0; y < potentialBalls.size(); y++ ){
        // Check if existed in last frame
        if(center.x - get<0>(potentialBalls[y]).x < 10 and center.x - get<0>(potentialBalls[y]).x > -10 and
        center.y - get<0>(potentialBalls[y]).y < 10 and center.y - get<0>(potentialBalls[y]).y > -10){
          get<2>(potentialBalls[y]) = get<2>(potentialBalls[y]) + 1;
          isNewPotentialBall = false;
        }
        // Check if existed sufficently long
        if(get<2>(potentialBalls[y]) > 3){
          get<0>(potentialBalls[y]) = get<0>(potentialBalls[y]) + maxPolyRectangle.tl();
          get<1>(potentialBalls[y])[0] = get<0>(potentialBalls[y]).x;
          get<1>(potentialBalls[y])[1] = get<0>(potentialBalls[y]).y;
          return potentialBalls[y];
        }
      }
      if(isNewPotentialBall){
        tuple<Point,Vec3f,int> supposedBall = make_tuple(center, circles[x], 0);
        potentialBalls.insert(potentialBalls.end(), supposedBall);
      }
    }
  }
  Point center(0,0);
  return make_tuple(center, NULL, 0);
};

Rect2d ballBound(Vec3f &ball){
  int radius = cvRound(ball[2]);
  Point topLeft = Point(cvRound(ball[0])-radius, cvRound(ball[1])-radius);
  Rect2d ballRectangle(topLeft.x, topLeft.y, radius*2, radius*2);
  return ballRectangle;
}

void drawBall(Mat &frame, Rect2d ballRectangle){
  rectangle(frame, ballRectangle, Scalar(0,255,0), 2, 8, 0);
}

void drawPerson(Mat &frame, Rect maxPolyRectangle){
  rectangle(frame, maxPolyRectangle.tl(), maxPolyRectangle.br(), Scalar(255,0,0), 2, 8, 0);
}

float calculateDiff(int cur, int prev){
  return ((float)cur-(float)prev);
}

float calculateWindow(queue<float> window){
  float sum = 0;
  while(window.size() > 0){
    sum = sum + window.front();
    window.pop();
  }
  return sum/(float) window.size();
}

/* We monitor the position of the ball as it's moving up and down. Sudden, drastic, changes in position
we ignore. If the center of the new box doesn't fall into the box in the previous frame it's unlikely to
be right. We continue logging in the window the position right afterward until the next significant change.
We don't know when the tracker gets back to tracking the ball again since two concurrent drastic changes doesn't
tell us if the tracker corrected itself or if it jumped to another incorrect location.

We create a window first for position and then using position create a window for velocity. The velocity window
elimanates large outliers using the method above. Our velocity window is still noisy. The tracker, while sufficently
accurate, still tends to be erratic. While the movement of the ball tends to a general direction it's constantly making
small adjustments counter to it's overall movement. For this we apply smoothing. Meaning we take the weighted
average of the window (using an applied kernel). Using these "smooth" value we can monitor a change in direction of velocity.
The means monitoring when velocity goes from + to - or from - to +. You can choose to smooth position or velocity
*/
void populateWindow(Rect2d newBox, Rect2d oldBox,queue<float> &smooth,
  queue<int> &acceleration, queue<float> &velocity, queue<float> &position,
  int maxBandwidth, int whatToSmooth){
  Point newCenter = Point(newBox.x+newBox.width/2, newBox.y+newBox.height/2);
  Point oldCenter = Point(oldBox.x+oldBox.width/2, oldBox.y+oldBox.height/2);
  if(oldBox.contains(newCenter)){
    position.push(newCenter.y);

    if(position.size() > 1){
      velocity.push(calculateDiff(newCenter.y,oldCenter.y));
    }
    if(velocity.size() > 1){
      acceleration.push(calculateDiff(newCenter.y,oldCenter.y));
    }

    if(position.size() > maxBandwidth){
      position.pop();
    }
    if(velocity.size() > maxBandwidth){
      velocity.pop();
    }
    if(acceleration.size() > maxBandwidth){
      acceleration.pop();
    }

    if(position.size() > 0 && whatToSmooth == 1){
      smooth.push(calculateWindow(position));
    }
    else if(velocity.size() > 0 && whatToSmooth == 2){
      smooth.push(calculateWindow(velocity));
    }
    else if(acceleration.size() > 0 && whatToSmooth == 3){
      smooth.push(calculateWindow(acceleration));
    }
    if(smooth.size() > maxBandwidth){
      smooth.pop();
    }
  }
}

list<float> kernel(int bandwidth, int kernelType){
  list<float> weight = new list<float>();
  switch(kernelType){
    case 1 :

    case 2 :

    case 3 :
  }
}

int main (int argc, const char * argv[])
{
    VideoCapture stream("/home/vanya/Videos/test_improved_downsample.avi"); // open the default camera (0) or file path
    VideoCapture capture("/home/vanya/Videos/test_improved_downsample.avi");
    if(!stream.isOpened())  // check if we succeeded
        return -1;

    // Frame Layers
    Mat frame;
    Mat still;
    Mat mask;
    Mat foreground;
    Mat background;
    Mat grey;

    // Feet and Ball
    tuple<Point,Vec3f,int> ball;

    // Verticle Position and Verticle Velocity
    queue<int> position = queue<float>();
    queue<float> velocity = queue<float>();
    queue<float> smooth = queue<float>();
    queue<float> acceleration = queue<float>();

    Rect2d ballRectangle;
    Rect2d ballRectangleOld;

    // Background Subtraction Settings
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    backgroundSubtraction(pMOG2);

    // Set up Trackers
    // Options are MIL, BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    Ptr<Tracker> ballTracker = Tracker::create("TLD");

    // Frame data
    vector<tuple<Point,Vec3f,int>> potentialBalls;

    // flags
    bool tracking = false;

    for(;;){
      // Grab Frame
      stream >> frame;

      // Convert it to Blurred & Grey
      cvtColor(frame,grey,CV_BGR2GRAY);
      GaussianBlur(grey,grey,Size(9,9),2,2);

      // Background Subtraction
      pMOG2->apply(grey, mask, 0);
      pMOG2->getBackgroundImage(background);

      // Clean Noise on Mask
      clean(mask);

      // Find Person
      Rect maxPolyRectangle = findPerson(mask);

      if(maxPolyRectangle.area() > 0 and !tracking){
        // Find Ball
        ball = findBall(grey, maxPolyRectangle, potentialBalls);
        ballRectangle = ballBound(get<1>(ball));

        if(get<2>(ball)){
          // Draw the Found Ball
          drawBall(frame, ballRectangle);

          // Track Ball
          ballTracker->init(frame, ballRectangle);

          // Set Tracking Flag
          tracking = true;

          imshow("Found", frame);
        }
      }

      if(tracking)
      {
        // Make Copy of Previous Track Box
        ballRectangleOld = Rect(ballRectangle.x, ballRectangle.y,
          ballRectangle.width, ballRectangle.height);

        // Update Track Box
        ballTracker->update(frame, ballRectangle);

        // Update Verticle Ball Movmement Data
        populateWindow(
          ballRectangle, ballRectangleOld,
          acceleration, smooth, velocity, position, 20, 1);

        //cout << acceleration.front() << endl;
        cout << smooth.front() << endl;

        // if(){
        //   capture.set(1,stream.get(CV_CAP_PROP_POS_FRAMES)-10);
        //   capture >> still;
        //   imshow("touch", still);
        //   }

        // Draw the tracked ball
        drawBall(frame, ballRectangle);
      }

      imshow("frame", frame);
      if(waitKey(1) >= 0) break;
    }

    // The camera will be deinitialized automatically in VideoCapture destructor

    return EXIT_SUCCESS;
}
