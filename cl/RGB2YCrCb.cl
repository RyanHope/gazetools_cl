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
__kernel void RGB2YCrCb(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  float4 rgb = read_imagef(srcImg, pos);
  float Ey = (0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
  float Ecb = .5f + (-0.169f * rgb.x - 0.331f * rgb.y + 0.500f * rgb.z);
  float Ecr = .5f + (0.500f * rgb.x - 0.419f * rgb.y - 0.081f * rgb.z);
  write_imagef(dstImg, pos, (float4)(Ey, Ecr, Ecb, 0));
}
