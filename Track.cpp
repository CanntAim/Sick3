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

int mode(vector<Mat> &hist, int channel){
  int bins = 255;
  double max = 0;
  double mode = 0;

  for(int i = 0; i < bins-1; i++){
    if(hist[channel].at<int>(i) > max){
      max=hist[channel].at<int>(i);
      mode = i;
    }
  }

  return floor(mode);
}

void filter(Mat &img, int color[3][3], int range){
  cout << to_string(color[0][0]) << endl;
  cout << to_string(color[0][1]) << endl;
  cout << to_string(color[0][2]) << endl;
  for(int i=0; i<img.rows; i++){
    for(int j=0; j<img.cols; j++){
      if((img.at<Vec3b>(i,j)[0] > (color[0][0] + range) || img.at<Vec3b>(i,j)[0] < (color[0][0] - range))
      && (img.at<Vec3b>(i,j)[1] > (color[0][1] + range) || img.at<Vec3b>(i,j)[1] < (color[0][1] - range))
      && (img.at<Vec3b>(i,j)[2] > (color[0][2] + range) || img.at<Vec3b>(i,j)[2] < (color[0][2] - range))
      && (img.at<Vec3b>(i,j)[0] > (color[1][0] + range) || img.at<Vec3b>(i,j)[0] < (color[1][0] - range))
      && (img.at<Vec3b>(i,j)[1] > (color[1][1] + range) || img.at<Vec3b>(i,j)[1] < (color[1][1] - range))
      && (img.at<Vec3b>(i,j)[2] > (color[1][2] + range) || img.at<Vec3b>(i,j)[2] < (color[1][2] - range))
      && (img.at<Vec3b>(i,j)[0] > (color[2][0] + range) || img.at<Vec3b>(i,j)[0] < (color[2][0] - range))
      && (img.at<Vec3b>(i,j)[1] > (color[2][1] + range) || img.at<Vec3b>(i,j)[1] < (color[2][1] - range))
      && (img.at<Vec3b>(i,j)[2] > (color[2][2] + range) || img.at<Vec3b>(i,j)[2] < (color[2][2] - range))){
        img.at<Vec3b>(i,j)[0] = 0;
        img.at<Vec3b>(i,j)[1] = 0;
        img.at<Vec3b>(i,j)[2] = 0;
      }
    }
  }
}

vector<Mat> histogram(Mat &img){
  int bins = 256;             // number of bins
  int nc = img.channels();    // number of channels

  vector<Mat> hist(nc);       // histogram arrays

  // Initalize histogram arrays
  for (int i = 0; i < hist.size(); i++)
    hist[i] = Mat::zeros(1, bins, CV_32SC1);

  // Calculate the histogram of the image
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      for (int k = 0; k < nc; k++){
	uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
	if(val != 0){
	  hist[k].at<int>(val) += 1;
        }
      }
    }
  }

  // For each histogram arrays, obtain the maximum (peak) value
  // Needed to normalize the display later
  int hmax[3] = {0,0,0};
  for (int i = 0; i < nc; i++){
    for (int j = 0; j < bins-1; j++){
      hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }
  }

  const char* wname[3] = { "blue", "green", "red" };
  Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

  vector<Mat> canvas(nc);

  // Display each histogram in a canvas
  for (int i = 0; i < nc; i++){
    canvas[i] = Mat::ones(125, bins, CV_8UC3);
    for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
      line(canvas[i],
	   Point(j, rows),
	   Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
	   nc == 1 ? Scalar(200,200,200) : colors[i], 1, 8, 0);
    }
    imshow(nc == 1 ? "value" : wname[i], canvas[i]);
  }
  return hist;
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
    approxPolyDP(cv::Mat(contours[i]), contoursPoly[i], 3, true);
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

int main (int argc, const char * argv[])
{
    VideoCapture stream("/home/vanya/Videos/Sick3/test_improved_downsample.avi"); // open the default camera (0) or file path
    VideoCapture capture("/home/vanya/Videos/Sick3/test_improved_downsample.avi");
    if(!stream.isOpened())  // check if we succeeded
        return -1;

    // Frame Layers
    Mat frame;
    Mat mask;
    Mat foreground;
    Mat background;
    Mat grey;

    // Exposure Layers
    Mat still;
    Mat blend;

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
    int modes[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

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

          // Calculate Primary Color of Ball and Feet
          Mat ballImage = frame(ballRectangle).clone();
          Mat leftFootImage = frame(get<0>(feet)).clone();
          Mat rightFootImage = frame(get<1>(feet)).clone();
          clean(ballImage);
          clean(leftFootImage);
          clean(rightFootImage);
          vector<Mat> hist = histogram(ballImage);
          modes[0][0] = mode(hist, 0);
          modes[0][1] = mode(hist, 1);
          modes[0][2] = mode(hist, 2);
          hist = histogram(leftFootImage);
          modes[1][0] = mode(hist, 0);
          modes[1][1] = mode(hist, 1);
          modes[1][2] = mode(hist, 2);
          hist = histogram(rightFootImage);
          modes[2][0] = mode(hist, 0);
          modes[2][1] = mode(hist, 1);
          modes[2][2] = mode(hist, 2);
          //mshow("histogram",hist);
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
            filter(blend,modes,5);
            imwrite( "/home/vanya/Pictures/Sick3/"+to_string(countTouch)+".jpg", blend);
          }
          lastTouch = stream.get(CV_CAP_PROP_POS_FRAMES);
          capture.set(1,lastTouch);
          capture >> still;
          blend = still.clone();
          countTouch++;
        } else if(countTouch > 0) {
          capture >> still;
          blend += still - blend;
          imshow(to_string(countTouch), blend);
        }
      }

      imshow("frame", frame);
      if(waitKey(1) >= 0) break;
    }

    // The camera will be deinitialized automatically in VideoCapture destructor

    return EXIT_SUCCESS;
}
