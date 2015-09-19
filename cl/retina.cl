#include "_subtended_angle.cl"

__kernel void blendmap(__constant float* ce, __constant float* var, float cx, float cy, float rx, float ry, float sw, float sh, float ez, float ex, float ey, unsigned int levels, __write_only image2d_t map)
{
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  float4 pixel = 0.f;

  pixel.x = _subtended_angle(x, y, cx, cy, rx, ry, sw, sh, ez, ex, ey);

  unsigned int l;
  for (l=0;l<=levels;l++) {
    if (pixel.x >= ce[l] && pixel.x < ce[l+1]) break;
  }

  if (l==0) {
    pixel.y = 1.0f;
    pixel.z = 1.0f;
    pixel.w = 1.0f;
  } else if (l==levels) {
    pixel.y = 1.0f/(1<<(levels-1));
    pixel.z = 0.0f;
    pixel.w = levels-1.0f;
  } else {
    float hres = 1.0f/(1<<(l-1));
    float lres = 1.0f/(1<<l);
    pixel.y = (lres*(pixel.x-ce[l]) + hres*(ce[l+1]-pixel.x)) / (ce[l+1]-ce[l]);
    float t1 = exp(-pixel.y*pixel.y/(2.0f*var[l]*var[l]));
    float t2 = exp(-pixel.y*pixel.y/(2.0f*var[l-1]*var[l-1]));
    pixel.z = (0.5f-t1)/(t2-t1);
    if ( pixel.z < 0 ) pixel.z = 0.0f;
    if ( pixel.z > 1 ) pixel.z = 1.0f;
    pixel.w = l;
  }

  write_imagef(map, (int2)(x, y), pixel);
}

__kernel void blend(__read_only image3d_t pyramid, __read_only image2d_t blendmap, unsigned int xoffset, unsigned int yoffset, __write_only image2d_t out)
{
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  float4 bm = read_imagef(blendmap, (int2)(x+xoffset,y+yoffset));

  float4 v1 = read_imagef(pyramid, (int4)(x,y,bm.w,0));
  float4 v2 = read_imagef(pyramid, (int4)(x,y,bm.w-1,0));

  float4 pixel = v1*(1-bm.z) + v2*bm.z;

  write_imagef(out, (int2)(x, y), pixel);
}
