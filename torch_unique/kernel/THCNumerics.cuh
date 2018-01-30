template <typename T>
struct ThrustEQOp {
  __device__ bool operator()(const T& a, const T& b) const { return a == b; }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct ThrustEQOp<half> {
  __device__ bool operator()(const half& a, const half& b) const {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __heq(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa == fb;
#endif
#else
    return THC_half2float(a) == THC_half2float(b);
#endif
  }
};
#endif

template <typename T>
struct ThrustLTOp {
  __device__ bool operator()(const T& a, const T& b) const { return a < b; }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct ThrustLTOp<half> {
  __device__ bool operator()(const half& a, const half& b) const {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hlt(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa < fb;
#endif
#else
    return THC_half2float(a) < THC_half2float(b);
#endif
  }
};
#endif
