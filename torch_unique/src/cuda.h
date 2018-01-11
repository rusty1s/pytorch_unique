void unique_single_cuda_Float (THCudaTensor*       input);
void unique_single_cuda_Double(THCudaDoubleTensor* input);
void unique_single_cuda_Byte  (THCudaByteTensor*   input);
void unique_single_cuda_Char  (THCudaCharTensor*   input);
void unique_single_cuda_Short (THCudaShortTensor*  input);
void unique_single_cuda_Int   (THCudaIntTensor*    input);
void unique_single_cuda_Long  (THCudaLongTensor*   input);

void unique_byKey_cuda_Float (THCudaTensor*       key, THCudaTensor*       value);
void unique_byKey_cuda_Double(THCudaDoubleTensor* key, THCudaDoubleTensor* value);
void unique_byKey_cuda_Byte  (THCudaByteTensor*   key, THCudaByteTensor*   value);
void unique_byKey_cuda_Char  (THCudaCharTensor*   key, THCudaCharTensor*   value);
void unique_byKey_cuda_Short (THCudaShortTensor*  key, THCudaShortTensor*  value);
void unique_byKey_cuda_Int   (THCudaIntTensor*    key, THCudaIntTensor*    value);
void unique_byKey_cuda_Long  (THCudaLongTensor*   key, THCudaLongTensor*   value);
