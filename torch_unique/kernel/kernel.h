#ifdef __cplusplus
extern "C" {
#endif

void unique_single_kernel_Float (THCState *state, THCudaTensor*       input);
void unique_single_kernel_Double(THCState *state, THCudaDoubleTensor* input);
void unique_single_kernel_Byte  (THCState *state, THCudaByteTensor*   input);
void unique_single_kernel_Char  (THCState *state, THCudaCharTensor*   input);
void unique_single_kernel_Short (THCState *state, THCudaShortTensor*  input);
void unique_single_kernel_Int   (THCState *state, THCudaIntTensor*    input);
void unique_single_kernel_Long  (THCState *state, THCudaLongTensor*   input);

void unique_byKey_kernel_Float (THCState *state, THCudaTensor*       key, THCudaTensor*       value);
void unique_byKey_kernel_Double(THCState *state, THCudaDoubleTensor* key, THCudaDoubleTensor* value);
void unique_byKey_kernel_Byte  (THCState *state, THCudaByteTensor*   key, THCudaByteTensor*   value);
void unique_byKey_kernel_Char  (THCState *state, THCudaCharTensor*   key, THCudaCharTensor*   value);
void unique_byKey_kernel_Short (THCState *state, THCudaShortTensor*  key, THCudaShortTensor*  value);
void unique_byKey_kernel_Int   (THCState *state, THCudaIntTensor*    key, THCudaIntTensor*    value);
void unique_byKey_kernel_Long  (THCState *state, THCudaLongTensor*   key, THCudaLongTensor*   value);

#ifdef __cplusplus
}
#endif
