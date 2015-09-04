__kernel void rgba2yuv(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  float4 rgb = read_imagef(srcImg, pos);
  write_imagef(dstImg, pos, rgb);
}
