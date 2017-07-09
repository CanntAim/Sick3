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

tuple<Rect2d,Rect2d> findFeet(tuple<Point,Vec3f,int> &ball){
  float radius = get<1>(ball)[2];
  Point rightFoot(cvRound(get<1>(ball)[0] + radius), cvRound(get<1>(ball)[1] - 0.5*radius));
  Point leftFoot(cvRound(get<1>(ball)[0] - 2*radius), cvRound(get<1>(ball)[1] - 0.5*radius));
  Rect2d rightFootRectangle(rightFoot.x, rightFoot.y, radius, radius*2);
  Rect2d leftFootRectangle(leftFoot.x, leftFoot.y, radius, radius*2);
  return make_tuple(leftFootRectangle,rightFootRectangle);
}

tuple<Point,Vec3f,int> findBall(Mat &grey, Rect &maxPolyRectangle, vector<tuple<Point,Vec3f,int>> &potentialBalls){
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

void drawFeet(Mat &frame, Rect2d &leftFoot, Rect2d &rightFoot){
  // Left Foot Rectangle
  rectangle(frame, leftFoot, Scalar(0,255,0), 2, 8, 0);
  // Right Foot Rectangle
  rectangle(frame, rightFoot, Scalar(0,255,0), 2, 8, 0);
}

void drawPerson(Mat &frame, Rect &maxPolyRectangle){
  // Person Rectangle
  rectangle(frame, maxPolyRectangle.tl(), maxPolyRectangle.br(), Scalar(255,0,0), 2, 8, 0);
}

int main (int argc, const char * argv[])
{
    VideoCapture stream("/test_improved_downsample.avi"); // open the default camera (0) or file path
    if(!stream.isOpened())  // check if we succeeded
        return -1;

    // Frame Layers
    Mat frame;
    Mat mask;
    Mat foreground;
    Mat background;
    Mat grey;
    Mat crop;

    // Background Subtraction Settings
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    backgroundSubtraction(pMOG2);

    // Set up Trackers
    // Options are MIL, BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    Ptr<Tracker> ballTracker = Tracker::create("MIL");
    Ptr<Tracker> leftFootTracker = Tracker::create("MIL");
    Ptr<Tracker> rightFootTracker = Tracker::create("MIL");

    // Frame data
    vector<tuple<Point,Vec3f,int>> potentialBalls;

    // flags
    bool tracking = false;

    for(;;){
      // Feet and Ball
      tuple<Rect2d,Rect2d> feet;
      tuple<Point,Vec3f,int> ball;
      Rect2d ballRectangle;
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
      crop = Mat(frame,maxPolyRectangle);

      if(maxPolyRectangle.area() > 0){
        // Find Ball
        ball = findBall(grey, maxPolyRectangle, potentialBalls);
        ballRectangle = ballBound(get<1>(ball));

        if(get<2>(ball)){
          // Find Feet
          feet = findFeet(ball);

          // Draw the found ball
          drawBall(crop, ballRectangle);

          // Draw the found feet
          drawFeet(crop, get<0>(feet), get<1>(feet));

          // Track Ball
          ballTracker->init(crop, ballRectangle);

          // Track Left Foot
          leftFootTracker->init(crop, get<0>(feet));

          // Track Right Foot
          rightFootTracker->init(crop, get<1>(feet));

          // Set Tracking Flag
          tracking = true;

          imshow("Found", crop);
        }
      }

      if(tracking)
      {
        // Update tracking results
        ballTracker->update(crop, ballRectangle);
        leftFootTracker->update(crop, get<0>(feet));
        rightFootTracker->update(crop, get<1>(feet));

        // Draw the tracked ball
        drawBall(crop, ballRectangle);

        // Draw the tracked feet
        drawFeet(crop, get<0>(feet), get<1>(feet));

        // Display result
        imshow("Tracking", crop);
      }

      imshow("frame", frame);
      if(waitKey(1) >= 0) break;
    }
    // The camera will be deinitialized automatically in VideoCapture destructor

    //Print the GRT version
    cout << "GRT Version: " << GRTBase::getGRTVersion() << endl;
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    return EXIT_SUCCESS;
}
