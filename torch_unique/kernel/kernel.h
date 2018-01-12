#ifdef __cplusplus
extern "C" {
#endif

void unique_kernel_Float (THCState *state, THCudaTensor*       input);
void unique_kernel_Double(THCState *state, THCudaDoubleTensor* input);
void unique_kernel_Byte  (THCState *state, THCudaByteTensor*   input);
void unique_kernel_Char  (THCState *state, THCudaCharTensor*   input);
void unique_kernel_Short (THCState *state, THCudaShortTensor*  input);
void unique_kernel_Int   (THCState *state, THCudaIntTensor*    input);
void unique_kernel_Long  (THCState *state, THCudaLongTensor*   input);

void uniqueByKey_kernel_Float (THCState *state, THCudaTensor*       key, THCudaTensor*       value);
void uniqueByKey_kernel_Double(THCState *state, THCudaDoubleTensor* key, THCudaDoubleTensor* value);
void uniqueByKey_kernel_Byte  (THCState *state, THCudaByteTensor*   key, THCudaByteTensor*   value);
void uniqueByKey_kernel_Char  (THCState *state, THCudaCharTensor*   key, THCudaCharTensor*   value);
void uniqueByKey_kernel_Short (THCState *state, THCudaShortTensor*  key, THCudaShortTensor*  value);
void uniqueByKey_kernel_Int   (THCState *state, THCudaIntTensor*    key, THCudaIntTensor*    value);
void uniqueByKey_kernel_Long  (THCState *state, THCudaLongTensor*   key, THCudaLongTensor*   value);

#ifdef __cplusplus
}
#endif
