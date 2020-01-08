//blackScholesAnalyticEngineKernelsCpu.cuh
//Scott Grauer-Gray
//Declarations of kernels for running black scholes using the analytic engine

#ifndef BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CPU_H
#define BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CPU_H

#include <stdlib.h>
#include <math.h>
#include <omp.h>

//needed for the structs used on the code
//#include "blackScholesAnalyticEngineStructs.h"
#include "blackScholesAnalyticEngineFuncs.h"

//global function to retrieve the output value for an option
void getOutValOptionCpu(optionInputStruct * options,
                        float * outputVals,
                        int numVals);

void getOutValOptionOpenMP(optionInputStruct * options,
                           float * outputVals,
                           int numVals);

#endif //BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CPU_H
