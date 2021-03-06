#include "_distance_2_point.cl"

__kernel void blend(__global unsigned char* pyramid, unsigned int n, __global float* blendmap, __global unsigned int* lmap, __global unsigned char* out, unsigned int w, unsigned int h, unsigned int x, unsigned int y)
{
  unsigned int c = get_global_id(0);
  unsigned int r = get_global_id(1);

  unsigned int bx = (w+1)/2-x;
  unsigned int by = (h+1)/2-y;
  unsigned int j = (r+by)*w + (c+bx);
  unsigned int l = lmap[j];
  float b = blendmap[j];

  unsigned int k = r*w + c;
  float v1 = pyramid[l*n + k];
  float v2 = pyramid[(l-1)*n + k];

  out[k] = (v1*(1-b) + v2*b);
}
