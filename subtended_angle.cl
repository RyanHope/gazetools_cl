#include "_distance_2_point.cl"

__kernel void subtended_angle(__constant float* x1, __constant float* y1, __constant float* x2, __constant float* y2, float rx, float ry, float sw, float sh, __constant float* ez, __constant float* ex, __constant float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  float d1 = _distance_2_point(x1[i], y1[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
  float d2 = _distance_2_point(x2[i], y2[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
  float dX = sw * (x2[i] - x1[i]) / rx;
  float dY = sh * (y2[i] - y1[i]) / ry;
  float dS = sqrt(dX*dX + dY*dY);
  float w1 = d1*d1 + d2*d2 - dS*dS;
  float w2 = 2 * d1 * d2;
  out[i] = acos(min(max(w1/w2, (float)-1.0), (float)1.0)) * 180.0 / M_PI;
}
