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

// __kernel void convolve1d_naive(__global float* in, __global float* out, __constant float* kernel_, unsigned int kernel__length, unsigned int halflen)
// {
//   unsigned int i = get_global_id(0);
//   unsigned int ii = i + halflen;
//   out[i] = in[ii] * kernel_[halflen];
//   for (unsigned int k=1; k<=halflen; k++) {
//     out[i] += in[ii+k] * kernel_[halflen-k] + in[ii-k] * kernel_[halflen+k];
//   }
// }

__kernel void convolve1d_naive(__read_only  image2d_t imgSrc,
                               __write_only image2d_t imgConvolved,
                               __constant   float *   kernelValues,
                                            int       w)
{
  // This kernel expects imgSrc to the mirror padded to the half kernel length,
  // imgConvolved should be the same dimensions as the original unpadded image.

  int x = get_global_id(0);
  int y = get_global_id(1);

  float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  for (int i = 0; i < w; i++)
  {
    convPix += read_imagef(imgSrc, (int2)(x+i,y)) * kernelValues[i];
  }
  write_imagef(imgConvolved, (int2)(x, y), convPix);
}
