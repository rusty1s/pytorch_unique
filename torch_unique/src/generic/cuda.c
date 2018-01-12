#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void unique_()(THCTensor *input) {
  return unique_kernel_()(state, input);
}

void unique_(ByKey)(THCTensor *key, THCTensor *value) {
  return unique_kernel_(ByKey)(state, key, value);
}

#endif
