#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

THCTensor *unique_kernel(THCState *state, THCudaLongTensor *index, THCTensor *input) {
  /* index = THCudaLongTensor_newContiguous(state, index); */
  input = THCTensor_(newContiguous)(state, input);
  THCTensor *output = input;

  thrust::device_ptr<real> output_data(THCTensor_(data)(state, input));
  ptrdiff_t size = THCTensor_(nElement)(state, output);

  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort, output_data, output_data + size);

  THCTensor_(free)(state, input);

  return NULL;
  /* return output; */
}

#endif
