// http://www.equasys.de/colorconversion.html
__kernel void YCbCr2RGB(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 yuv = read_imageui(srcImg, pos);
  float Ey = yuv.x;
  float Ecb = yuv.y - 128;
  float Ecr = yuv.z - 128;
  uchar R = 1.000f * Ey + 0.000f * Ecb + 1.400f * Ecr;
  uchar G = 1.000f * Ey - 0.343f * Ecb - 0.711f * Ecr;
  uchar B = 1.000f * Ey + 1.765f * Ecb + 0.000f * Ecr;

  write_imageui(dstImg, pos, (uint4)(R, G, B, 0));
}
