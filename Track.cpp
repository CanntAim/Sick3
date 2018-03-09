#include "Track.h"

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
  Rect personRectangle;
  findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
  int personRectangleArea = 0;
  vector<Rect> bounds(contours.size());
  vector<vector<Point>> contoursPoly(contours.size());
  for(int i = 0; i < contours.size(); i++){
    approxPolyDP(contours[i], contoursPoly[i], 3, true);
    for(int i = 0; i < contoursPoly.size(); i++){
      if(boundingRect(contoursPoly[i]).area() > personRectangleArea){
        personRectangle = boundingRect(contoursPoly[i]);
        personRectangleArea = personRectangle.area();
      }
    }
  }
  return personRectangle;
}

tuple<Rect2d,Rect2d> findFeet(tuple<Point,Vec3f,int> &ball){
  float radius = get<1>(ball)[2];
  Point rightFoot(cvRound(get<1>(ball)[0] + radius), cvRound(get<1>(ball)[1] - 0.5*radius));
  Point leftFoot(cvRound(get<1>(ball)[0] - 2*radius), cvRound(get<1>(ball)[1] - 0.5*radius));
  Rect2d rightFootRectangle(rightFoot.x, rightFoot.y, radius, radius*2);
  Rect2d leftFootRectangle(leftFoot.x, leftFoot.y, radius, radius*2);
  return make_tuple(leftFootRectangle,rightFootRectangle);
}


tuple<Point,Vec3f,int> findBall(Mat &grey, Rect personRectangle, vector<tuple<Point,Vec3f,int>> &potentialBalls){
  vector<Vec3f> circles;
  HoughCircles(Mat(grey,personRectangle), circles, CV_HOUGH_GRADIENT, 1, grey.rows/8,200,25,0,100);

  for( size_t x = 0; x < circles.size(); x++ ){
    double heightThreshold = (double)(personRectangle.height)/5.0;
    if (circles[x][1] > personRectangle.height - heightThreshold){
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
          get<0>(potentialBalls[y]) = get<0>(potentialBalls[y]) + personRectangle.tl();
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

void drawBallTrace(Mat &frame, int frameCount, Rect2d ballRectangle){
  Point ballCenter = Point(
    ballRectangle.x+ballRectangle.width/2,
    ballRectangle.y+ballRectangle.height/2);
  circle(frame, ballCenter, 10, Scalar(0,0,255), -1, 8);
}

void drawBall(Mat &frame, Rect2d ballRectangle){
  rectangle(frame, ballRectangle, Scalar(0,255,0), 2, 8, 0);
}

void drawPerson(Mat &frame, Rect personRectangle){
  rectangle(frame, personRectangle.tl(), personRectangle.br(), Scalar(255,0,0), 2, 8, 0);
}

void drawFeet(Mat &frame, Rect2d &leftFoot, Rect2d &rightFoot){
  // Left Foot Rectangle
  rectangle(frame, leftFoot, Scalar(0,255,0), 2, 8, 0);
  // Right Foot Rectangle
  rectangle(frame, rightFoot, Scalar(0,255,0), 2, 8, 0);
}

int calculateDifference(int cur, int prev){
  return (cur-prev);
}

float calculateWindow(deque<int> window, vector<float> weights){
  int index = 0;
  float value = 0;
  while(window.size() && weights.size() > index){
    value = value + (float)window.front()*weights[index];
    window.pop_front();
    index++;
  }
  return value;
}

void populateWindow(Rect2d newBox, Rect2d oldBox,
  deque<float> &smooth,
  deque<int> &acceleration, deque<int> &velocity, deque<int> &position,
  vector<float> weights, int toSmooth){
  Point newCenter = Point(newBox.x+newBox.width/2, newBox.y+newBox.height/2);
  Point oldCenter = Point(oldBox.x+oldBox.width/2, oldBox.y+oldBox.height/2);
  int oldVelocity = 0;
  int newVelocity = 0;
  if(oldBox.contains(newCenter)){
    position.push_back(newCenter.y);
    if(position.size() > 1){
      oldVelocity = velocity.front();
      velocity.push_back(calculateDifference(newCenter.y, oldCenter.y));
      newVelocity = velocity.front();
    }
    if(velocity.size() > 1){
      acceleration.push_back(calculateDifference(newVelocity, oldVelocity));
    }

    if(position.size() >= weights.size()){
      position.pop_front();
    }
    if(velocity.size() >= weights.size()){
      velocity.pop_front();
    }
    if(acceleration.size() >= weights.size()){
      acceleration.pop_front();
    }

    if(position.size() > 0 && toSmooth == 1){
      smooth.push_back(calculateWindow(position, weights));
    }
    else if(velocity.size() > 0 && toSmooth == 2){
      smooth.push_back(calculateWindow(velocity, weights));
    }
    else if(acceleration.size() > 0 && toSmooth == 3){
      smooth.push_back(calculateWindow(acceleration, weights));
    }
    if(smooth.size() > weights.size()){
      smooth.pop_front();
    }
  }
}

vector<float> kernel(vector<int> weights, int bandwidth){
  vector<float> slopes;
  vector<float> calculatedWeights;
  size_t windowIndex = 0;
  size_t slopeIndex = -1;
  for(size_t i = 0; i < weights.size()-1; i++){
    slopes.push_back(weights[i+1]-weights[i]);
  }
  calculatedWeights.push_back(weights[0]);
  while((int)windowIndex < (int)bandwidth){
    int slopeSize = slopes.size();
    int remain = windowIndex % (bandwidth/slopeSize);
    if(remain == 0){
      slopeIndex++;
    }
    calculatedWeights.push_back(calculatedWeights.back() +
    slopes[slopeIndex]/(bandwidth/slopeSize));
    windowIndex++;
  }

  int sumCalculatedWeights = 0;
  for(size_t i = 0; i < calculatedWeights.size(); i++){
    sumCalculatedWeights = sumCalculatedWeights + calculatedWeights[i];
  }
  for(size_t i = 0; i < calculatedWeights.size(); i++){
    calculatedWeights[i] = calculatedWeights[i]/sumCalculatedWeights;
  }
  return calculatedWeights;
}

bool checkDirectionChange(deque<float> &buffer, int touch, int frame, int difference){
  if(buffer.size() > 1){
    return (buffer.at(0) > 0 && buffer.at(1) < 0
    && (frame - touch) > difference);
  }
}

bool checkDribbling(bool &flag, int verticalPostion, Rect personRectangle){
  double heightThreshold = (double)(personRectangle.height)/5.0;
  double verticalThreshold = (double)personRectangle.y
  + (double)personRectangle.height
  - heightThreshold;
  if((double)verticalPostion < verticalThreshold){
    flag = true;
  }
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
			   double, const Scalar& color){
  for(int y = 0; y < cflowmap.rows; y += step){
    for(int x = 0; x < cflowmap.cols; x += step){
      const Point2f& fxy = flow.at<Point2f>(y, x);
      line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
      circle(cflowmap, Point(x,y), 2, color, -1);
    }
  }
}


void trace(VideoCapture &stream, Mat &still,
	   Mat &grey, Mat &prevgrey,
	   Mat &flow, Mat &uflow, Mat &cflow){
  stream >> still;
  cvtColor(still, grey, CV_BGR2GRAY);	  
  if(!prevgrey.empty()) {
    calcOpticalFlowFarneback(prevgrey, grey, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cvtColor(prevgrey, cflow, CV_GRAY2BGR);
    uflow.copyTo(flow);
    drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
  }
}

int main (int argc, const char * argv[])
{
    VideoCapture stream("/home/vanya/Videos/Sick3/test_improved_downsample.avi"); // open the default camera (0) or file path
    if(!stream.isOpened())  // check if we succeeded
        return -1;

    // Frame Layers
    Mat frame;
    Mat mask;
    Mat foreground;
    Mat background;
    Mat grey, prevgrey;
    Mat flow, cflow, uflow;

    // Result Layers
    Mat still;

    // Kernel Weights
    vector<int> weights;
    weights.push_back(0);
    weights.push_back(3);
    weights.push_back(10);
    weights.push_back(3);
    weights.push_back(0);
    vector<float> normalizedWeights = kernel(weights, 20);

    // Feet and Ball
    tuple<Rect2d,Rect2d> feet;
    tuple<Point,Vec3f,int> ball;
    Rect2d ballRectangle;
    Rect2d ballRectangleOld;

    // Verticle Position and Verticle Velocity
    deque<int> acceleration = deque<int>();
    deque<int> position = deque<int>();
    deque<int> velocity = deque<int>();
    deque<float> smooth = deque<float>();

    // Background Subtraction Settings
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    backgroundSubtraction(pMOG2);

    // Set up Trackers
    // Options are MIL, BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    Ptr<Tracker> ballTracker = TrackerTLD::create();

    // Frame data
    vector<tuple<Point,Vec3f,int>> potentialBalls;

    // Flags
    bool tracking = false;
    bool dribbling = false;

    // Meta-data
    int lastTouch = 0;
    int countTouch = 0;
    int frameCycleTouch = 0;

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
      Rect personRectangle = findPerson(mask);

      if(personRectangle.area() > 0 and !tracking){
        // Find Ball
        ball = findBall(grey, personRectangle, potentialBalls);
        ballRectangle = ballBound(get<1>(ball));

        if(get<2>(ball)){
          // Find Feet
          feet = findFeet(ball);

          // Draw the Found Ball
          drawBall(frame, ballRectangle);

          // Draw the found feet
          drawFeet(frame, get<0>(feet), get<1>(feet));

          // Track Ball
          ballTracker->init(frame, ballRectangle);

          // Set Tracking Flag
          tracking = true;

	  // Display Found
          imshow("Found", frame);
        }
      }

      if(tracking){
        // Make Copy of Previous Track Box
        ballRectangleOld = Rect(ballRectangle.x, ballRectangle.y,
          ballRectangle.width, ballRectangle.height);

        // Update Track Box
        ballTracker->update(frame, ballRectangle);

        // Update Verticle Ball Movmement Data
        populateWindow(
          ballRectangle, ballRectangleOld,
          smooth, acceleration, velocity, position, normalizedWeights, 2);

        // Check if dribbling yet
        checkDribbling(dribbling, position.front(), personRectangle);

        // Draw the tracked ball
        drawBall(frame, ballRectangle);
      }

      if(tracking && dribbling){
        if(checkDirectionChange(smooth, lastTouch, stream.get(CV_CAP_PROP_POS_FRAMES), 10)){
          if(countTouch > 0){
	    imwrite( "/home/vanya/Pictures/Sick3/"+to_string(countTouch)+".jpg", cflow);
          }
          lastTouch = stream.get(CV_CAP_PROP_POS_FRAMES);
          trace(stream, still, grey, prevgrey, flow, uflow, cflow);
          frameCycleTouch = 0;
          countTouch++;
        } else if(countTouch > 0) {
          trace(stream, still, grey, prevgrey, flow, uflow, cflow);
          drawBallTrace(cflow, frameCycleTouch, ballRectangle);
          imshow(to_string(countTouch), cflow);
          frameCycleTouch++;
        }
      }

      imshow("frame", frame);
      if(waitKey(1) >= 0) break;
      std::swap(prevgrey, grey);
    }

    // The camera will be deinitialized automatically in VideoCapture destructor

    return EXIT_SUCCESS;
}
