#include "_distance_2_point.cl"

__kernel void resmap(__global float* ecc, unsigned int n, __constant float* ce, unsigned int nl, __constant float* var, __global float* resmap, __global float* blendmap, __global unsigned int* lmap)
{
  unsigned int i = get_global_id(0);
  unsigned int l;
  for (l=0;l<=nl;l++) {
    if (ecc[i] >= ce[l] && ecc[i] < ce[l+1]) break;
  }
  if (l==0) {
    resmap[i] = 1.0f;
    blendmap[i] = 1.0f;
    lmap[i] = 1;
  } else if (l==nl) {
    resmap[i] = 1.0f/(1<<(nl-1));
    blendmap[i] = 0.0f;
    lmap[i] = nl-1;
  } else {
    float hres = 1.0f/(1<<(l-1));
    float lres = 1.0f/(1<<l);
    resmap[i] = (lres*(ecc[i]-ce[l]) + hres*(ce[l+1]-ecc[i])) / (ce[l+1]-ce[l]);
    float t1 = exp(-resmap[i]*resmap[i]/(2.0f*var[l]*var[l]));
		float t2 = exp(-resmap[i]*resmap[i]/(2.0f*var[l-1]*var[l-1]));
    blendmap[i] = (0.5f-t1)/(t2-t1);
		if ( blendmap[i] < 0 ) blendmap[i] = 0.0f;
		if ( blendmap[i] > 1 ) blendmap[i] = 1.0f;
    lmap[i] = l;
  }
}
