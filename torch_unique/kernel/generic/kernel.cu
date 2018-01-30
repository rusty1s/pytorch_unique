#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void unique_()(THCState *state, THCTensor *input) {
  input = THCTensor_(newContiguous)(state, input);

  thrust::device_ptr<real> first(THCTensor_(data)(state, input));
  ptrdiff_t numel = THCTensor_(nElement)(state, input);

  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort, first, first + numel, ThrustLTOp<real>());
  thrust::device_ptr<real> last(THRUST_EXEC(thrust::unique, first, first + numel, ThrustEQOp<real>()));

  numel = last - first;
  THCTensor_(resize1d)(state, input, numel);

  THCTensor_(free)(state, input);
}

void unique_(ByKey)(THCState *state, THCTensor *key, THCTensor *value) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, key, value));

  key = THCTensor_(newContiguous)(state, key);
  value = THCTensor_(newContiguous)(state, value);

  thrust::device_ptr<real> firstKey(THCTensor_(data)(state, key));
  thrust::device_ptr<real> firstValue(THCTensor_(data)(state, value));
  ptrdiff_t numel = THCTensor_(nElement)(state, key);

  THRUST_ALLOC(state);
  THRUST_EXEC(thrust::sort_by_key, firstKey, firstKey + numel, firstValue, ThrustLTOp<real>());
  thrust::pair<thrust::device_ptr<real>, thrust::device_ptr<real> > last(THRUST_EXEC(thrust::unique_by_key, firstKey, firstKey + numel, firstValue, ThrustEQOp<real>()));

  THCTensor_(resize1d)(state, key, last.first - firstKey);
  THCTensor_(resize1d)(state, value, last.second - firstValue);

  THCTensor_(free)(state, key);
  THCTensor_(free)(state, value);
}

#endif
