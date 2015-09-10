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

#include "_distance_2_point.cl"

__kernel void distance_2_point(__global float* x, __global float* y, float rx, float ry, float sw, float sh, __global float* ez, __global float* ex, __global float* ey, __global float* out)
{
  unsigned int i = get_global_id(0);
  out[i] = _distance_2_point(x[i], y[i], rx, ry, sw, sh, ez[i], ex[i], ey[i]);
}
