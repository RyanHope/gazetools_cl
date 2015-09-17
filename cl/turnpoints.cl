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

__kernel void turnpoints(__global float* src, unsigned int span, float threshold, __global float* out)
{
  unsigned int i = get_global_id(0);
  unsigned int s = i + span;
  out[i] = 0;
  unsigned int count = 0;
  for (unsigned int j=1; j<=span; j++) {
    if (src[s]>src[s+j] && src[s]>src[s-j])
      count++;
    else
      break;
  }
  if (count==span)
    out[i] = 1;
  else {
    count = 0;
    for (unsigned int j=1; j<=span; j++) {
      if (src[s]<src[s+j] && src[s]<src[s-j])
        count++;
      else
        break;
    }
    if (count==span)
      out[i] = -1;
  }
}
