#include "subdiv2D_index.h"

Subdiv2DIndex::Subdiv2DIndex(cv::Rect rectangle) : cv::Subdiv2D{rectangle} {}

void Subdiv2DIndex::getTrianglesIndices(
    std::vector<cv::Vec3i>& traingleIndexes) const {
  traingleIndexes.clear();
  int i, total = (int)(qedges.size() * 4);
  std::vector<bool> edgemask(total, false);
  const bool filterPoints = true;
  cv::Rect2f rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x,
                  bottomRight.y - topLeft.y);

  for (i = 4; i < total; i += 2) {
    if (edgemask[i])
      continue;
    cv::Point2f a, b, c;
    int edge_a = i;
    int indexA = edgeOrg(edge_a, &a) - 4;
    if (filterPoints && !rect.contains(a))
      continue;
    int edge_b = getEdge(edge_a, NEXT_AROUND_LEFT);
    int indexB = edgeOrg(edge_b, &b) - 4;
    if (filterPoints && !rect.contains(b))
      continue;
    int edge_c = getEdge(edge_b, NEXT_AROUND_LEFT);
    int indexC = edgeOrg(edge_c, &c) - 4;
    if (filterPoints && !rect.contains(c))
      continue;
    edgemask[edge_a] = true;
    edgemask[edge_b] = true;
    edgemask[edge_c] = true;

    cv::Vec3i traingle = cv::Vec3i(indexA, indexB, indexC);

    traingleIndexes.push_back(traingle);
  }
}