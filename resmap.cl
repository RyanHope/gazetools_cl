#include "_distance_2_point.cl"

__kernel void resmap(__global float* ecc, __constant float* ce, unsigned int nl)
{
  unsigned int i = get_global_id(0);
  unsigned int l = 0;
  while (l<=nl) {
    if (ecc[i] >= ce[l] && ecc[i] < ce[l+1]) {
      if (l==0) {
        ecc[i] = 1.0f;
      } else if (l==nl) {
        ecc[i] = 1.0f/(1<<(nl-1));
      } else {
        float hres = 1.0f/(1<<(l-1));
        float lres = 1.0f/(1<<l);
        ecc[i] = (lres*(ecc[i]-ce[l]) + hres*(ce[l+1]-ecc[i])) / (ce[l+1]-ce[l]);
      }
      return;
    }
    l++;
  }
}
