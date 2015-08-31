#include "_subtended_angle.cl"

__kernel void subtended_angle(__constant float* x1, __constant float* y1, __constant float* x2, __constant float* y2, float rx, float ry, float sw, float sh, __constant float* ez, __constant float* ex, __constant float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _subtended_angle(x1[i], y1[i], x2[i], y2[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
