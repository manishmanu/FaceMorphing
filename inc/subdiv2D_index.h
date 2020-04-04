#pragma once

#include <opencv4/opencv2/imgproc.hpp>

class Subdiv2DIndex : public cv::Subdiv2D {
 public:
  Subdiv2DIndex(cv::Rect rectangle);

  // Source code of Subdiv2D:
  // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/subdivision2d.cpp#L762
  // The implementation tweaks getTrianglesList() so that only the indice of the
  // triangle inside the image are returned
  void getTrianglesIndices(std::vector<cv::Vec3i>& traingleIndexes) const;
};