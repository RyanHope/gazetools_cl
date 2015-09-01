#include "_distance_2_point.cl"

inline float _subtended_angle(float x1, float y1, float x2, float y2, float rx, float ry, float sw, float sh, float ez, float ex, float ey)
{
  float d1 = _distance_2_point(x1, y1, rx, ry, sw, sh, ez, ex, ey);
  float d2 = _distance_2_point(x2, y2, rx, ry, sw, sh, ez, ex, ey);
  float dX = sw * (x2 - x1) / rx;
  float dY = sh * (y2 - y1) / ry;
  float dS = sqrt(dX*dX + dY*dY);
  float w1 = d1*d1 + d2*d2 - dS*dS;
  float w2 = 2.0f * d1 * d2;
  return acos(min(max(w1/w2, -1.0f), 1.0f)) * 180.0f / M_PI_F;
}
