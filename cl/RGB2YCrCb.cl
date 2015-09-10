// http://www.equasys.de/colorconversion.html
__kernel void RGB2YCrCb(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  float4 rgb = read_imagef(srcImg, pos);
  float Ey = (0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
  float Ecb = .5f + (-0.169f * rgb.x - 0.331f * rgb.y + 0.500f * rgb.z);
  float Ecr = .5f + (0.500f * rgb.x - 0.419f * rgb.y - 0.081f * rgb.z);
  write_imagef(dstImg, pos, (float4)(Ey, Ecr, Ecb, 0));
}
