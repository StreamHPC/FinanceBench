//blackScholesAnalyticEngineKernels.cuh
//Scott Grauer-Gray
//Kernels for running black scholes using the analytic engine

#ifndef BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_H
#define BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_H

#include <stdio.h>

#include <stdlib.h>

#include <vector>
#include <algorithm>

#include <hip/hip_runtime.h>
//needed for the structs used on the code
#include "blackScholesAnalyticEngineStructs.h"


//constants used in this code
#define M_1_SQRTPI  0.564189583547756286948
#define M_SQRT_2    0.7071067811865475244008443621048490392848359376887

//global function to retrieve the output value for an option
__global__ void getOutValOption(optionInputStruct * options,
                                float * outputVals,
                                int numVals);

__global__ void getOutValOptionOpt(optionInputStruct * options,
                                   float * outputVals,
                                   int numVals);

__global__ void getOutValOptionOpt(char * type,
                                   float * data,
                                   float * outputVals,
                                   int numVals);

#endif //BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_H
