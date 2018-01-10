#include <cstddef>

class THCThrustAllocator {
  public:
    typedef char value_type;

    THCThrustAllocator(THCState* state) : state_(state) {
    }

    ~THCThrustAllocator() {
    }

    char* allocate(std::ptrdiff_t size) {
      char* out = NULL;
      THCudaCheck(THCudaMalloc(state_, (void**) &out, size));
      return out;
    }

    void deallocate(char* p, size_t size) {
      THCudaCheck(THCudaFree(state_, p));
    }

  private:
    THCState* state_;
};
