// http://www.equasys.de/colorconversion.html
__kernel void YCrCb2RGB(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 yuv = read_imageui(srcImg, pos);
  float Ey = yuv.x;
  float Ecr = yuv.y;
  float Ecb = yuv.z;
  uchar R = 1.000f * Ey + 0.000f * (Ecb-128) + 1.400f * (Ecr-128);
  uchar G = 1.000f * Ey - 0.343f * (Ecb-128) - 0.711f * (Ecr-128);
  uchar B = 1.000f * Ey + 1.765f * (Ecb-128) + 0.000f * (Ecr-128);
  write_imageui(dstImg, pos, (uint4)(R, G, B, 0));
}
