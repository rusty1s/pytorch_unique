void unique_single_cuda_Float (THCudaTensor*       input);
void unique_single_cuda_Double(THCudaDoubleTensor* input);
void unique_single_cuda_Byte  (THCudaByteTensor*   input);
void unique_single_cuda_Char  (THCudaCharTensor*   input);
void unique_single_cuda_Short (THCudaShortTensor*  input);
void unique_single_cuda_Int   (THCudaIntTensor*    input);
void unique_single_cuda_Long  (THCudaLongTensor*   input);

void unique_byKey_cuda_Float (TTHCudaTensor*       key, THCudaTensor*       value);
void unique_byKey_cuda_Double(TTHCudaDoubleTensor* key, THCudaDoubleTensor* value);
void unique_byKey_cuda_Byte  (TTHCudaByteTensor*   key, THCudaByteTensor*   value);
void unique_byKey_cuda_Char  (TTHCudaCharTensor*   key, THCudaCharTensor*   value);
void unique_byKey_cuda_Short (TTHCudaShortTensor*  key, THCudaShortTensor*  value);
void unique_byKey_cuda_Int   (TTHCudaIntTensor*    key, THCudaIntTensor*    value);
void unique_byKey_cuda_Long  (TTHCudaLongTensor*   key, THCudaLongTensor*   value);
