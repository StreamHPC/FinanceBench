//monteCarloKernels.cuh
//Scott Grauer-Gray
//May 10, 2012
//Kernel headers for running monte carlo on the GPU

#ifndef MONTE_CARLO_KERNELS_CUH
#define MONTE_CARLO_KERNELS_CUH

//needed for constants related to monte carlo
#include "monteCarloConstants.h"

//needed for structs related to monte carlo
#include "monteCarloStructs.h"

//needed for curand
#include <curand_kernel.h>

#define A_1 -39.696830286653757
#define A_2 220.94609842452050
#define A_3 -275.92851044696869
#define A_4 138.35775186726900
#define A_5 -30.664798066147160
#define A_6 2.5066282774592392
#define B_1 -54.476098798224058
#define B_2 161.58583685804089
#define B_3 -155.69897985988661
#define B_4 66.801311887719720
#define B_5 -13.280681552885721

//function to set up the random states
__global__ void setup_kernel(curandState * state,
                             int seedVal,
                             int numSamples);

//function to compute the inverse normal distribution
__device__ dataType compInverseNormDist(dataType x);

__device__ dataType interestRateCompoundFact(dataType t,
                                             dataType rate);

__device__ dataType interestRateDiscountFact(dataType t,
                                             dataType rate);

__device__ dataType flatForwardDiscountImpl(dataType t,
                                            dataType rate);

__device__ dataType yieldTermStructDiscount(dataType t,
                                            dataType rate);

__device__ dataType interestRateImpliedRate(dataType compound,
                                            dataType t);

__device__ dataType yieldTermStructForwardRate(dataType t1,
                                               dataType t2,
                                               dataType rate);

__device__ dataType processDrift(dataType t,
                                 dataType x,
                                 monteCarloOptionStruct optionStruct);

__device__ dataType discretizationDrift(dataType t0,
                                        dataType x0,
                                        dataType dt,
                                        monteCarloOptionStruct optionStruct);

__device__ dataType localVoltLocVol(dataType t,
                                    dataType underlyingLevel,
                                    monteCarloOptionStruct optionStruct);

__device__ dataType processDiff(dataType t,
                                dataType x,
                                monteCarloOptionStruct optionStruct);

__device__ dataType discDiff(dataType t0,
                             dataType x0,
                             dataType dt,
                             monteCarloOptionStruct optionStruct);

__device__ dataType stdDeviation(dataType t0,
                                 dataType x0,
                                 dataType dt,
                                 monteCarloOptionStruct optionStruct);

__device__ dataType apply(dataType x0,
                          dataType dx);

__device__ dataType discDrift(dataType t0,
                              dataType x0,
                              dataType dt,
                              monteCarloOptionStruct optionStruct);

__device__ dataType processEvolve(dataType t0,
                                  dataType x0,
                                  dataType dt,
                                  dataType dw,
                                  monteCarloOptionStruct optionStruct);

//retrieve the current sequence
__device__ void getSequence(dataType * sequence,
                            dataType sampleNum);

__device__ dataType getProcessValX0(monteCarloOptionStruct optionStruct);

__device__ void getPath(dataType * path,
                        size_t sampleNum,
                        dataType dt,
                        curandState * state,
                        monteCarloOptionStruct optionStruct);

__device__ dataType getPrice(dataType val);

//initialize the path
__device__ void initializePath(dataType * path);

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    curandState * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int numSamples);

__device__ void getPath(dataType * path,
                        size_t sampleNum,
                        dataType dt,
                        curandStatePhilox4_32_10_t * state,
                        monteCarloOptionStruct optionStruct);

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    curandStatePhilox4_32_10_t * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples);

#endif //MONTE_CARLO_KERNELS_CUH
