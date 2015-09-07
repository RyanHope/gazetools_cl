__kernel void convolve1d_naive(__gloabl float* in, __gloabl float* out, int length, float* kernel_, int kernel__length)
{
  unsigned int i = get_global_id(0);
  for(int k=0; k<kernel__length; k++){
    out[i] += in[i+k] * kernel_[kernel__length - k - 1];
  }
}
