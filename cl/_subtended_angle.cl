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

inline float _subtended_angle(float x1, float y1, float x2, float y2, float rx, float ry, float sw, float sh, float ez, float ex, float ey)
{
  float d1 = _distance_2_point(x1, y1, rx, ry, sw, sh, ez, ex, ey);
  float d2 = _distance_2_point(x2, y2, rx, ry, sw, sh, ez, ex, ey);
  float dX = sw * (x2 - x1) / rx;
  float dY = sh * (y2 - y1) / ry;
  float dS = sqrt(dX*dX + dY*dY);
  float w1 = d1*d1 + d2*d2 - dS*dS;
  float w2 = 2.0f * d1 * d2;
  return acos(min(max(w1/w2, -1.0f), 1.0f)) * 180.0f / M_PI_F;
}
