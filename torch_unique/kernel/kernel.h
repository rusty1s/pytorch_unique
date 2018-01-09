#ifdef __cplusplus
extern "C" {
#endif

THCudaTensor*       unique_kernel_Float (THCState *state, THCudaLongTensor* index, THCudaTensor*       input);
THCudaDoubleTensor* unique_kernel_Double(THCState *state, THCudaLongTensor* index, THCudaDoubleTensor* input);
THCudaByteTensor*   unique_kernel_Byte  (THCState *state, THCudaLongTensor* index, THCudaByteTensor*   input);
THCudaCharTensor*   unique_kernel_Char  (THCState *state, THCudaLongTensor* index, THCudaCharTensor*   input);
THCudaShortTensor*  unique_kernel_Short (THCState *state, THCudaLongTensor* index, THCudaShortTensor*  input);
THCudaIntTensor*    unique_kernel_Int   (THCState *state, THCudaLongTensor* index, THCudaIntTensor*    input);
THCudaLongTensor*   unique_kernel_Long  (THCState *state, THCudaLongTensor* index, THCudaLongTensor*   input);

#ifdef __cplusplus
}
#endif
