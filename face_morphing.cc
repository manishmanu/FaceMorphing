#include "face_morphing.h"

#include "face_landmark_detection.h"
#include "subdiv2D_index.h"

FaceMorphing::FaceMorphing() {}

// Draw delaunay triangles
void FaceMorphing::draw_delaunay(cv::Mat img,
                                 std::vector<cv::Point2f> points,
                                 std::vector<cv::Vec3i> triangles) {
  cv::Rect rect(0, 0, img.size().width, img.size().height);
  cv::Scalar color = cv::Scalar(0, 255, 0);
  for (int i = 0; i < triangles.size(); i++) {
    cv::Point2f a = points[triangles[i][0]];
    cv::Point2f b = points[triangles[i][1]];
    cv::Point2f c = points[triangles[i][2]];
    line(img, a, b, color, 2, 8, 0);
    line(img, b, c, color, 2, 8, 0);
    line(img, c, a, color, 2, 8, 0);
  }

  cv::imshow("delaunay traingles", img);
  cv::waitKey(0);
}

void FaceMorphing::drawPoints(cv::Mat img, std::vector<cv::Point2f> points) {
  for (int i = 0; i < points.size(); i++) {
    cv::circle(img, points[i], 3, cv::Scalar(0, 255, 0), -1, 8, 0);
  }
  cv::imshow("face landmark points", img);
  cv::waitKey(0);
}

bool checkBoundary(cv::Point2f point, cv::Size size) {
  if (point.x < 0 || point.x >= size.width) {
    return false;
  }
  if (point.y < 0 || point.y >= size.height) {
    return false;
  }
  return true;
}

void FaceMorphing::getDelaunyTrianglesIndexes(
    cv::Size imageSize,
    std::vector<cv::Point2f> points,
    std::vector<cv::Vec3i>& triangleIndexes) {
  cv::Rect rect(0, 0, imageSize.width, imageSize.height);

  Subdiv2DIndex subdivIndex(rect);

  for (int i = 0; i < points.size(); i++) {
    if (checkBoundary(points[i], imageSize)) {
      subdivIndex.insert(points[i]);
    } else {
      std::cout
          << "ERROR points given for delauny triangulation are out of boundary"
          << std::endl;
      return;
    }
  }

  subdivIndex.getTrianglesIndices(triangleIndexes);
}

/*
Derived from:
https://github.com/spmallick/learnopencv/tree/master/FaceMorph
https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
*/
// Apply affine transform calculated using srcTri and dstTri to src
void FaceMorphing::applyAffineTransform(cv::Mat& warpImage,
                                        cv::Mat& src,
                                        std::vector<cv::Point2f>& srcTri,
                                        std::vector<cv::Point2f>& dstTri) {
  // Given a pair of triangles, find the affine transform.
  cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);

  // Apply the Affine Transform just found to the src image
  cv::warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR,
                 cv::BORDER_REFLECT_101);
}

/*
Derived from:
https://github.com/spmallick/learnopencv/tree/master/FaceMorph
https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
*/
// Warps and alpha blends triangular regions from img1 and img2 to img
void FaceMorphing::morphTriangle(cv::Mat& img1,
                                 cv::Mat& img2,
                                 cv::Mat& img,
                                 std::vector<cv::Point2f>& t1,
                                 std::vector<cv::Point2f>& t2,
                                 std::vector<cv::Point2f>& t,
                                 double alpha) {
  // Find bounding rectangle for each triangle
  cv::Rect r = cv::boundingRect(t);
  cv::Rect r1 = cv::boundingRect(t1);
  cv::Rect r2 = cv::boundingRect(t2);

  // Offset points by left top corner of the respective rectangles
  std::vector<cv::Point2f> t1Rect, t2Rect, tRect;
  std::vector<cv::Point> tRectInt;
  for (int i = 0; i < 3; i++) {
    tRect.push_back(cv::Point2f(t[i].x - r.x, t[i].y - r.y));
    tRectInt.push_back(
        cv::Point(t[i].x - r.x, t[i].y - r.y));  // for fillConvexPoly

    t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
    t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
  }

  // Get mask by filling triangle
  cv::Mat mask = cv::Mat::zeros(r.height, r.width, CV_8UC3);
  cv::fillConvexPoly(mask, tRectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

  // Apply warpImage to small rectangular patches
  cv::Mat img1Rect, img2Rect;
  img1(r1).copyTo(img1Rect);
  img2(r2).copyTo(img2Rect);

  cv::Mat warpImage1 = cv::Mat::zeros(r.height, r.width, img1Rect.type());
  cv::Mat warpImage2 = cv::Mat::zeros(r.height, r.width, img2Rect.type());

  applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
  applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

  // Alpha blend rectangular patches
  cv::Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

  // Copy triangular region of the rectangular patch to the output image
  cv::multiply(imgRect, mask, imgRect);
  cv::multiply(img(r), cv::Scalar(1.0, 1.0, 1.0) - mask, img(r));
  img(r) = img(r) + imgRect;
}

cv::Mat FaceMorphing::morphFace(cv::Mat src, cv::Mat dest, double alpha) {
  if (src.size() != dest.size()) {
    std::cout << "WARN src and dest image sizes are not same !!!" << std::endl;
    if (src.size().area() > dest.size().area()) {
      std::cout << "INFO resizing src to be of same size as dest" << std::endl;
      cv::resize(src, src, dest.size(), cv::INTER_LINEAR);
    } else {
      std::cout << "INFO resizing dest to be of same size as src" << std::endl;
      cv::resize(dest, dest, src.size(), cv::INTER_LINEAR);
    }
  }

  cv::Mat imgMorph = cv::Mat::zeros(src.size(), CV_8UC3);

  std::vector<cv::Point2f> srcPoints =
      FaceLandmarkDetection::get().getFacialLandmarks(src);
  std::vector<cv::Point2f> destPoints =
      FaceLandmarkDetection::get().getFacialLandmarks(dest);
  std::vector<cv::Point2f> interimPoints;
  for (int i = 0; i < srcPoints.size(); i++) {
    float x, y;
    x = (1 - alpha) * srcPoints[i].x + alpha * destPoints[i].x;
    y = (1 - alpha) * srcPoints[i].y + alpha * destPoints[i].y;
    interimPoints.push_back(cv::Point2f(x, y));
  }

  // drawPoints(src, srcPoints);
  // drawPoints(dest, destPoints);

  // add corners and mid-corner points
  /* o.......o.......o
     .................
     .................
     o...............o
     .................
     .................
     o.......o.......o
  */
  std::vector<cv::Point2f> boundary_points;
  boundary_points.push_back(cv::Point2f(0, 0));
  boundary_points.push_back(cv::Point2f(src.cols / 2, 0));
  boundary_points.push_back(cv::Point2f(src.cols - 1, 0));
  boundary_points.push_back(cv::Point2f(src.cols - 1, src.rows / 2));
  boundary_points.push_back(cv::Point2f(src.cols - 1, src.rows - 1));
  boundary_points.push_back(cv::Point2f(src.cols / 2, src.rows - 1));
  boundary_points.push_back(cv::Point2f(0, src.rows - 1));
  boundary_points.push_back(cv::Point2f(0, src.rows / 2));

  for (int i = 0; i < boundary_points.size(); i++) {
    srcPoints.push_back(boundary_points[i]);
    destPoints.push_back(boundary_points[i]);
    interimPoints.push_back(boundary_points[i]);
  }

  std::vector<cv::Vec3i> triangles;
  getDelaunyTrianglesIndexes(src.size(), srcPoints, triangles);

  // draw_delaunay(src, srcPoints, triangles);
  // draw_delaunay(dest, destPoints, triangles);

  std::vector<cv::Point2f> src_t, dest_t, interim_t;
  for (int i = 0; i < triangles.size(); i++) {
    src_t.clear();
    dest_t.clear();
    interim_t.clear();
    for (int j = 0; j < 3; j++) {
      src_t.push_back(srcPoints[triangles[i][j]]);
      dest_t.push_back(destPoints[triangles[i][j]]);
      interim_t.push_back(interimPoints[triangles[i][j]]);
    }
    morphTriangle(src, dest, imgMorph, src_t, dest_t, interim_t, alpha);
  }

  return imgMorph;
}

void FaceMorphing::writeFaceMorphingVideo(cv::Mat src,
                                          cv::Mat dest,
                                          float rate) {
  // resizing images to match lowest area among two.
  if (src.size() != dest.size()) {
    std::cout << "WARN src and dest image sizes are not same !!!" << std::endl;
    if (src.size().area() > dest.size().area()) {
      std::cout << "INFO resizing src to be of same size as dest" << std::endl;
      cv::resize(src, src, dest.size(), cv::INTER_LINEAR);
    } else {
      std::cout << "INFO resizing dest to be of same size as src" << std::endl;
      cv::resize(dest, dest, src.size(), cv::INTER_LINEAR);
    }
  }
  cv::VideoWriter video("morphing_gif.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                        cv::Size(src.cols, src.rows));

  int static_frames = 20;
  float alpha = 0.f;
  // beginning
  for (int i = 0; i < static_frames; i++) {
    cv::Mat morphedImage = morphFace(src, dest, alpha);
    video.write(morphedImage);
  }

  for (; alpha <= 1.0; alpha = alpha + rate) {
    cv::Mat morphedImage = morphFace(src, dest, alpha);
    video.write(morphedImage);
  }

  for (int i = 0; i < static_frames; i++) {
    cv::Mat morphedImage = morphFace(src, dest, alpha);
    video.write(morphedImage);
  }

  video.release();
}