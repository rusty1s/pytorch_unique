#include <THC/THC.h>

#define unique_ TH_CONCAT_2(unique_cuda_, Real)

extern THCState *state;

#include "generic/cuda.c"
#include "THCGenerateFloatType.h"
#include "generic/cuda.c"
#include "THCGenerateDoubleType.h"
#include "generic/cuda.c"
#include "THCGenerateByteType.h"
#include "generic/cuda.c"
#include "THCGenerateCharType.h"
#include "generic/cuda.c"
#include "THCGenerateShortType.h"
#include "generic/cuda.c"
#include "THCGenerateIntType.h"
#include "generic/cuda.c"
#include "THCGenerateLongType.h"
