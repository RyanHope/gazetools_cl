//===============================================================================
// This file is part of gazetools.
// Copyright (C) 2015 Ryan M. Hope <rmh3093@gmail.com>
//
// gazetools is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// gazetools is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with gazetools.  If not, see <http://www.gnu.org/licenses/>.
//===============================================================================

// http://www.equasys.de/colorconversion.html
__kernel void YCrCb2RGB(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  float4 yuv = read_imagef(srcImg, pos);
  float R = 1.000f * yuv.x + 0.000f * (yuv.z-.5f) + 1.400f * (yuv.y-.5f);
  float G = 1.000f * yuv.x - 0.343f * (yuv.z-.5f) - 0.711f * (yuv.y-.5f);
  float B = 1.000f * yuv.x + 1.765f * (yuv.z-.5f) + 0.000f * (yuv.y-.5f);
  write_imagef(dstImg, pos, (float4)(R, G, B, 0));
}
