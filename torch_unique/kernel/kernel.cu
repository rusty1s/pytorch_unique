#include <THC.h>

#include "kernel.h"

#define unique_kernel TH_CONCAT_2(unique_kernel, Real)

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
#include "generic/kernel.cu"
#include "THCGenerateByteType.h"
#include "generic/kernel.cu"
#include "THCGenerateCharType.h"
#include "generic/kernel.cu"
#include "THCGenerateShortType.h"
#include "generic/kernel.cu"
#include "THCGenerateIntType.h"
#include "generic/kernel.cu"
#include "THCGenerateLongType.h"
