#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

class FaceMorphing {
 public:
  FaceMorphing();

  cv::Mat morphFace(cv::Mat src, cv::Mat dest, double alpha);

  // rate should be between 0 and 1. Rate is the progress rate of face morphing
  // from source to destination.
  void writeFaceMorphingVideo(cv::Mat src, cv::Mat dest, float rate);

 protected:
  void applyAffineTransform(cv::Mat& warpImage,
                            cv::Mat& src,
                            std::vector<cv::Point2f>& srcTri,
                            std::vector<cv::Point2f>& dstTri);
  void morphTriangle(cv::Mat& img1,
                     cv::Mat& img2,
                     cv::Mat& img,
                     std::vector<cv::Point2f>& t1,
                     std::vector<cv::Point2f>& t2,
                     std::vector<cv::Point2f>& t,
                     double alpha);

  void getDelaunyTrianglesIndexes(cv::Size imageSize,
                                  std::vector<cv::Point2f> points,
                                  std::vector<cv::Vec3i>& triangleIndexes);

  void drawPoints(cv::Mat img, std::vector<cv::Point2f> points);
  void draw_delaunay(cv::Mat img,
                     std::vector<cv::Point2f> points,
                     std::vector<cv::Vec3i> triangles);
};