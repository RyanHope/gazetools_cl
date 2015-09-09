#define BLOCK_SIZE 16

__kernel void BasicConvolve(__read_only  image2d_t imgSrc,
                            __global     float *   kernelValues,
                            __global     int *     kernelSize,
                            __write_only image2d_t imgConvolved)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   //Kernel size (ideally, odd number)
   //global_size should be [width-w/2, height-w/2]
   //Writes answer to [x+w/2, y+w/2]
   int w = kernelSize[0];

   int x = get_global_id(0);
   int y = get_global_id(1);

   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

   for (int i = 0; i < w; i++)
   {
       for (int j = 0; j < w; j++)
       {
          convPix += read_imagef(imgSrc, smp, (int2)(x+i,y+j)) * kernelValues[i + w*j];
       }
   }

  //  pix = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);
   write_imagef(imgConvolved, (int2)(x + (w>>1), y + (w>>1)), convPix);
}

__kernel void ConvolveConst(__read_only  image2d_t imgSrc,
                            __constant   float *   kernelValues,
                            __constant   int *     kernelSize,
                            __write_only image2d_t imgConvolved)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   //Kernel size (ideally, odd number)
   //global_size should be [width-w/2, height-w/2]
   //Writes answer to [x+w/2, y+w/2]
   int w = kernelSize[0];

   int x = get_global_id(0);
   int y = get_global_id(1);

   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
   float4 temp;
   uint4 pix;

   int2 coords;

   for (int i = 0; i < w; i++)
   {
       for (int j = 0; j < w; j++)
       {
          coords.x = x+i; coords.y = y+j;

          pix = read_imageui(imgSrc, smp, coords);
          temp = (float4)((float)pix.x, (float)pix.y, (float)pix.z, (float)pix.w);

          convPix += temp * kernelValues[i + w*j];
       }
   }

   coords.x = x + (w>>1); coords.y = y + (w>>1);
   pix = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);
   write_imageui(imgConvolved, coords, pix);
}

__kernel void ConvolveLocal(__read_only  image2d_t imgSrc,
                            __constant   float *   kernelValues,
                            __constant   int *     kernelSize,
                            __write_only image2d_t imgConvolved)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   int w = kernelSize[0];
   int wBy2 = w>>1; //w divided by 2

   //Goes up to 21x21 filters
    __local uint4 P[BLOCK_SIZE+20][BLOCK_SIZE+20];

    //Identification of this workgroup
   int i = get_group_id(0);
   int j = get_group_id(1);

   //Identification of work-item
   int idX = get_local_id(0);
   int idY = get_local_id(1);

   int ii = i*BLOCK_SIZE + idX; // == get_global_id(0);
   int jj = j*BLOCK_SIZE + idY; // == get_global_id(1);

   int2 coords = (int2)(ii, jj);

   //Reads pixels
   P[idX][idY] = read_imageui(imgSrc, smp, coords);

   //Needs to read extra elements for the filter in the borders
   if (idX < w)
   {
      coords.x = ii + BLOCK_SIZE; coords.y = jj;
      P[idX + BLOCK_SIZE][idY] = read_imageui(imgSrc, smp, coords);
   }

   if (idY < w)
   {
      coords.x = ii; coords.y = jj + BLOCK_SIZE;
      P[idX][idY + BLOCK_SIZE] = read_imageui(imgSrc, smp, coords);
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   //Computes convolution
   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
   float4 temp;

   for (int ix = 0; ix < w; ix++)
   {
       int tx = idX + ix;
       for (int jy = 0; jy < w; jy++)
       {
          int ty = idY + jy;
          temp = (float4)((float)P[tx][ty].x, (float)P[tx][ty].y, (float)P[tx][ty].z, (float)P[tx][ty].w);
          convPix += temp * kernelValues[ix + w*jy];
       }
   }

   P[idX+wBy2][idY+wBy2] = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);

   barrier(CLK_LOCAL_MEM_FENCE);
   coords = (int2)(ii+wBy2, jj+wBy2);
   write_imageui(imgConvolved, coords, P[idX+wBy2][idY+wBy2]);
}
