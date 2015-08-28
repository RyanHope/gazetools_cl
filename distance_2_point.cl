#include "_distance_2_point.cl"

__kernel void distance_2_point(__constant float* x, __constant float* y, float rx, float ry, float sw, float sh, __constant float* ez, __constant float* ex, __constant float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _distance_2_point(x[i], y[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
