#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include "drawLandmarks.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

int main(int argc, char** argv) {
  // Load Face Detector
  CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

  // Create an instance of Facemark
  Ptr<Facemark> facemark = FacemarkLBF::create();

  // Load landmark detector
  facemark->loadModel("lbfmodel.yaml");

  // Set up webcam for video capture
  //   VideoCapture cam(0);

  // Variable to store a video frame and its grayscale
  Mat frame, gray;

  frame = imread(
      "/home/manish/Personal/LearnOpenCV/myProjects/dlib/examples/faces/"
      "Tom_Cruise_avp_2014_4.jpg");

  // Read a frame
  while (true) {
    // Find face
    vector<Rect> faces;
    // Convert frame to grayscale because
    // faceDetector requires grayscale image.
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Detect faces
    faceDetector.detectMultiScale(gray, faces);

    // Variable for landmarks.
    // Landmarks for one face is a vector of points
    // There can be more than one face in the image. Hence, we
    // use a vector of vector of points.
    vector<vector<Point2f> > landmarks;

    // Run landmark detector
    bool success = facemark->fit(frame, faces, landmarks);

    if (success) {
      // If successful, render the landmarks on the face
      for (int i = 0; i < landmarks.size(); i++) {
        std::cout << "points size : " << landmarks[i].size() << std::endl;
        drawLandmarks(frame, landmarks[i]);
      }
    }

    // Display results
    cv::imshow("Facial Landmark Detection", frame);
    cv::waitKey(0);
    // Exit loop if ESC is pressed
    if (waitKey(1) == 27)
      break;
    break;
  }
  return 0;
}
