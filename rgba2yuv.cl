__kernel void rgba2yuv(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 rgb = read_imageui(srcImg, pos);
  write_imageui(dstImg, pos, rgb);
}
