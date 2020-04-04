#include "face_landmark_detection.h"

/*
Derived from:
https://github.com/spmallick/learnopencv/tree/master/FaceMorph
https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
*/
FaceLandmarkDetection::FaceLandmarkDetection() {
  faceDetector = cv::CascadeClassifier("../haarcascade_frontalface_alt2.xml");
  facemark = cv::face::FacemarkLBF::create();
  facemark->loadModel("../lbfmodel.yaml");
}

FaceLandmarkDetection& FaceLandmarkDetection::get() {
  static FaceLandmarkDetection instance;
  return instance;
}

std::vector<cv::Point2f> FaceLandmarkDetection::getFacialLandmarks(
    cv::Mat image) {
  cv::Mat gray;
  std::vector<cv::Point2f> landmarks;
  std::vector<cv::Rect> faces;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // Detect faces
  faceDetector.detectMultiScale(gray, faces);

  if (faces.size() > 1 || faces.size() == 0) {
    std::cout << "ERROR exactly one face should be present in the images. "
                 "faces detected : "
              << faces.size()
              << ". Doesn't support "
                 "Face Morphing"
              << std::endl;
    return landmarks;
  }

  std::vector<std::vector<cv::Point2f>> all_landmarks;
  // Run landmark detector
  bool success = facemark->fit(image, faces, all_landmarks);

  if (success) {
    if (all_landmarks.size() == 1) {
      landmarks = all_landmarks[0];
    }
  } else {
    std::cout << "ERROR facial landmark detection failed" << std::endl;
  }

  return landmarks;
}
