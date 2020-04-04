#include <iostream>
#include <memory>

#include "face_morphing.h"

int main() {
  cv::Mat ledge = cv::imread("../examples/ledge.jpg");
  cv::Mat joker = cv::imread("../examples/joker.png");

  std::shared_ptr<FaceMorphing> faceMorphing = std::make_shared<FaceMorphing>();
  faceMorphing->writeFaceMorphingVideo(ledge, joker, 0.015);

  cv::Mat morphed_image = faceMorphing->morphFace(ledge, joker, 0.5);
  cv::imwrite("morphed_image.png", morphed_image);
}