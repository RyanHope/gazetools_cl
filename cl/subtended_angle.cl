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

#include "_subtended_angle.cl"

__kernel void subtended_angle(__global float* x1, __global float* y1, __global float* x2, __global float* y2, float rx, float ry, float sw, float sh, __global float* ez, __global float* ex, __global float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _subtended_angle(x1[i], y1[i], x2[i], y2[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
