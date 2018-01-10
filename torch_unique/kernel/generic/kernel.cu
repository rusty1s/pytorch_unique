#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

THCTensor *unique_kernel(THCState *state, THCudaLongTensor *index, THCTensor *input) {
  input = THCTensor_(newContiguous)(state, input);

  thrust::device_ptr<real> idxThrust(THCTensor_(data)(state, input));
  ptrdiff_t numel = THCTensor_(nElement)(state, input);
  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort, idxThrust, idxThrust + numel);
  thrust::device_ptr<real> endIdxThrust(THRUST_EXEC(thrust::unique, idxThrust, idxThrust + numel));
  numel = endIdxThrust - idxThrust;
  THCTensor_(resize1d)(state, input, numel);

  THCTensor_(free)(state, input);

  return input;
}

#endif
