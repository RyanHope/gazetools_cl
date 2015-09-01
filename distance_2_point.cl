#include "_distance_2_point.cl"

__kernel void distance_2_point(__global float* x, __global float* y, float rx, float ry, float sw, float sh, __global float* ez, __global float* ex, __global float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _distance_2_point(x[i], y[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
