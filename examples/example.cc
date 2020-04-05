#include <iostream>
#include <memory>

#include "face_morphing.h"

int main() {
  cv::Mat decaprio1 = cv::imread("../examples/decaprio1.jpg");
  cv::Mat decaprio2 = cv::imread("../examples/decaprio2.jpg");

  std::shared_ptr<FaceMorphing> faceMorphing = std::make_shared<FaceMorphing>();
  faceMorphing->writeFaceMorphingVideo(decaprio1, decaprio2, 0.015);

  cv::Mat morphed_image = faceMorphing->morphFace(decaprio1, decaprio2, 0.5);
  cv::imwrite("morphed_image.png", morphed_image);
}