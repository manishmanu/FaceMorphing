#include <opencv4/opencv2/face.hpp>
#include <opencv4/opencv2/opencv.hpp>

class FaceLandmarkDetection {
 public:
  static FaceLandmarkDetection& get();

  std::vector<cv::Point2f> getFacialLandmarks(cv::Mat image);

 private:
  FaceLandmarkDetection();
  cv::Ptr<cv::face::Facemark> facemark;
  cv::CascadeClassifier faceDetector;
};