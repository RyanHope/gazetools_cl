// http://www.equasys.de/colorconversion.html
__kernel void RGB2YCrCb(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 rgb = read_imageui(srcImg, pos);
  float R = rgb.x;
  float G = rgb.y;
  float B = rgb.z;
  uchar Ey = 0 + (0.299f * R + 0.587f * G + 0.114f * B);
  uchar Ecb = 128 + (-0.169f * R - 0.331f * G + 0.500f * B);
  uchar Ecr = 128 + (0.500f * R - 0.419f * G - 0.081f * B);
  write_imageui(dstImg, pos, (uint4)(Ey, Ecr, Ecb, 0));
}
