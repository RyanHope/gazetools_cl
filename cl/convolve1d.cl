__kernel void convolve1d_naive(__global float* in, __global float* out, __constant float* kernel_, unsigned int kernel__length, unsigned int halflen)
{
  unsigned int i = get_global_id(0);
  for(int k=0; k<kernel__length; k++){
    out[i+halflen] += in[i+k] * kernel_[kernel__length - k - 1];
  }
}
