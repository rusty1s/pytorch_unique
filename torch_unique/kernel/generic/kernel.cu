#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void unique_(single)(THCState *state, THCTensor *input) {
  input = THCTensor_(newContiguous)(state, input);

  thrust::device_ptr<real> first(THCTensor_(data)(state, input));
  ptrdiff_t numel = THCTensor_(nElement)(state, input);
  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort, first, first + numel);
  thrust::device_ptr<real> last(THRUST_EXEC(thrust::unique, first, first + numel));
  numel = last - first;
  THCTensor_(resize1d)(state, input, numel);

  THCTensor_(free)(state, input);
}

void unique_(byKey)(single)(THCState *state, THCTensor *key, THCTensor *value) {
  key = THCTensor_(newContiguous)(state, key);
  value = THCTensor_(newContiguous)(state, value);

  thrust::device_ptr<real> firstKey(THCTensor_(data)(state, key));
  thrust::device_ptr<real> firstValue(THCTensor_(data)(state, value));
  ptrdiff_t numel = THCTensor_(nElement)(state, key);
  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort_by_key, firstKey, firstKey + numel, firstValue);
  thrust::device_ptr<real> last(THRUST_EXEC(thrust::unique_by_key, firstKey, firstKey + numel, firstValue));
  numel = last - first;
  THCTensor_(resize1d)(state, key, numel);
  THCTensor_(resize1d)(state, value, numel);

  THCTensor_(free)(state, key);
  THCTensor_(free)(state, value);
}

#endif
