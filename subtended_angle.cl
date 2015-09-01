#include "_subtended_angle.cl"

__kernel void subtended_angle(__global float* x1, __global float* y1, __global float* x2, __global float* y2, float rx, float ry, float sw, float sh, __global float* ez, __global float* ex, __global float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _subtended_angle(x1[i], y1[i], x2[i], y2[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
