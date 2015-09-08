__kernel void convolve1d_naive(__global float* in, __global float* out, __constant float* kernel_, unsigned int kernel__length, unsigned int halflen)
{
  unsigned int i = get_global_id(0);
  unsigned int ii = i + halflen;
  out[i] = in[ii] * kernel_[halflen];
  for (unsigned int k=1; k<=halflen; k++) {
    out[i] += in[ii+k] * kernel_[halflen-k] + in[ii-k] * kernel_[halflen+k];
  }
}
