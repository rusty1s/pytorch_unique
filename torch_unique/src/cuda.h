void unique_cuda_Float (THCudaTensor*       input);
void unique_cuda_Double(THCudaDoubleTensor* input);
void unique_cuda_Byte  (THCudaByteTensor*   input);
void unique_cuda_Char  (THCudaCharTensor*   input);
void unique_cuda_Short (THCudaShortTensor*  input);
void unique_cuda_Int   (THCudaIntTensor*    input);
void unique_cuda_Long  (THCudaLongTensor*   input);

void uniqueByKey_cuda_Float (THCudaTensor*       key, THCudaTensor*       value);
void uniqueByKey_cuda_Double(THCudaDoubleTensor* key, THCudaDoubleTensor* value);
void uniqueByKey_cuda_Byte  (THCudaByteTensor*   key, THCudaByteTensor*   value);
void uniqueByKey_cuda_Char  (THCudaCharTensor*   key, THCudaCharTensor*   value);
void uniqueByKey_cuda_Short (THCudaShortTensor*  key, THCudaShortTensor*  value);
void uniqueByKey_cuda_Int   (THCudaIntTensor*    key, THCudaIntTensor*    value);
void uniqueByKey_cuda_Long  (THCudaLongTensor*   key, THCudaLongTensor*   value);
