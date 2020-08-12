//monteCarloKernels.cu
//Scott Grauer-Gray
//May 10, 2012
//GPU Kernels for running monte carlo

#include "monteCarloKernelsGpu.h"

#define mmax(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

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

//function to compute the inverse normal distribution
__device__ inline dataType compInverseNormDist(dataType x)
{
    dataType z;
    //if (x < x_low_ || x_high_ < x) {
    //z = tail_value(x);
    //} else {

    z = x - 0.5;
    dataType r = z * z;
    z = (((((A_1 * r+ A_2) *r + A_3) * r + A_4) * r + A_5) * r + A_6) * z /
        (((((B_1 * r+ B_2) *r + B_3) * r + B_4) * r + B_5) * r + 1.0);

    return z;
}

__device__ inline dataType interestRateCompoundFact(dataType t,
                                                    dataType rate)
{
    //assuming "continuous" option
    return exp(rate * t);
}

__device__ inline dataType interestRateDiscountFact(dataType t,
                                                    dataType rate)
{
    return 1.0 / interestRateCompoundFact(t, rate);
}

__device__ inline dataType flatForwardDiscountImpl(dataType t,
                                                   dataType rate)
{
    return interestRateDiscountFact(t, rate);
}

__device__ inline dataType yieldTermStructDiscount(dataType t,
                                                   dataType rate)
{
    return flatForwardDiscountImpl(t, rate);
}

__device__ inline dataType interestRateImpliedRate(dataType compound,
                                                   dataType t)
{
    dataType r = log(compound) / t;
    return r;
}

__device__ inline dataType yieldTermStructForwardRate(dataType t1,
                                                      dataType t2,
                                                      dataType rate)
{
    dataType compound = interestRateDiscountFact(t1, rate) /
                        interestRateDiscountFact(t2, rate);
    return interestRateImpliedRate(compound, t2 - t1);
}

__device__ inline dataType localVoltLocVol(dataType t,
                                           dataType underlyingLevel,
                                           monteCarloOptionStruct optionStruct)
{
    (void) t;
    (void) underlyingLevel;
    return optionStruct.voltVal;
}

__device__ inline dataType processDiff(dataType t,
                                       dataType x,
                                       monteCarloOptionStruct optionStruct)
{
    return localVoltLocVol(t, x, optionStruct);
}

__device__ inline dataType processDrift(dataType t,
                                        dataType x,
                                        monteCarloOptionStruct optionStruct)
{
    dataType sigma = processDiff(t, x, optionStruct);
    dataType t1 = t + 0.0001;
    return yieldTermStructForwardRate(t, t1, optionStruct.riskVal) -
           yieldTermStructForwardRate(t, t1, optionStruct.divVal) -
           (0.5 * sigma * sigma);
}

__device__ inline dataType discretizationDrift(dataType t0,
                                               dataType x0,
                                               dataType dt,
                                               monteCarloOptionStruct optionStruct)
{
    return processDrift(t0, x0, optionStruct) * dt;
}

__device__ inline dataType discDiff(dataType t0,
                                    dataType x0,
                                    dataType dt,
                                    monteCarloOptionStruct optionStruct)
{
    return processDiff(t0, x0, optionStruct) * sqrt(dt);
}

__device__ inline dataType stdDeviation(dataType t0,
                                        dataType x0,
                                        dataType dt,
                                        monteCarloOptionStruct optionStruct)
{
    return discDiff(t0, x0, dt, optionStruct);
}

__device__ inline dataType apply(dataType x0,
                                 dataType dx)
{
    return (x0 * exp(dx));
}

__device__ inline dataType discDrift(dataType t0,
                                     dataType x0,
                                     dataType dt,
                                     monteCarloOptionStruct optionStruct)
{
    return processDrift(t0, x0, optionStruct) * dt;
}

__device__ inline dataType processEvolve(dataType t0,
                                         dataType x0,
                                         dataType dt,
                                         dataType dw,
                                         monteCarloOptionStruct optionStruct)
{
    return apply(
        x0, discDrift(t0, x0, dt, optionStruct) +
        stdDeviation(t0, x0, dt, optionStruct) * dw
    );
}

//retrieve the current sequence
__device__ inline void getSequence(dataType * sequence,
                                   dataType sampleNum)
{
    (void) sampleNum;
    for(unsigned int iInSeq = 0; iInSeq < SEQUENCE_LENGTH; ++iInSeq)
    {
        sequence[iInSeq] = DEFAULT_SEQ_VAL;
    }
}

__device__ inline dataType getProcessValX0(monteCarloOptionStruct optionStruct)
{
    return optionStruct.underlyingVal;
}

__device__ inline dataType getPrice(dataType val)
{
    return mmax(STRIKE_VAL - val, 0.0f) * DISCOUNT_VAL;
}

//initialize the path
__device__ inline void initializePath(dataType * path)
{
    for(unsigned int i = 0; i < SEQUENCE_LENGTH; ++i)
    {
        path[i] = START_PATH_VAL;
    }
}

//function to set up the random states
__global__ void setup_kernel(hiprandState * state,
                             int seedVal,
                             int numSamples)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numSamples)
    {
        /* Each thread gets same seed , a different sequence
        number , no offset */
        hiprand_init(seedVal, id, 0, &(state[id]));
    }
}

__device__ inline void getPathV1(dataType * path,
                                 size_t sampleNum,
                                 dataType dt,
                                 hiprandState * state,
                                 monteCarloOptionStruct optionStruct)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        dataType randVal = hiprand_uniform(&(state[sampleNum]));
        dataType inverseCumRandVal = compInverseNormDist(randVal);
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

__device__ inline void getPathV2(dataType * path,
                                 size_t sampleNum,
                                 dataType dt,
                                 hiprandState * state,
                                 monteCarloOptionStruct optionStruct)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        //dataType randVal = curand_uniform(&(state[sampleNum]));
        //dataType inverseCumRandVal = compInverseNormDist(randVal);
        dataType inverseCumRandVal = hiprand_normal(&(state[sampleNum]));
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

//template<>
__device__ inline void getPath(dataType * path,
                               size_t sampleNum,
                               dataType dt,
                               hiprandStatePhilox4_32_10_t * state,
                               monteCarloOptionStruct optionStruct)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        //dataType randVal = curand_uniform(&(state[sampleNum]));
        //dataType inverseCumRandVal = compInverseNormDist(randVal);
        dataType inverseCumRandVal = hiprand_normal(&(state[sampleNum]));
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    hiprandState * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples)
{
    (void) times;
    (void) seedVal;

    //retrieve the thread number
    size_t numThread = blockIdx.x * blockDim.x + threadIdx.x;

    //retrieve the option number
    int numOption = 0;

    //retrieve the number of sample
    int numSample = numThread;

    size_t outputNum = numSample;

    //while (numSample < numSamples)
    if(numSample < numSamples)
    {
        #if defined(__HCC__)
        hiprand_init(seedVal, numSample, 0, &(state[numSample]));
        #endif
        //declare and initialize the path
        dataType path[SEQUENCE_LENGTH];
        initializePath(path);

        #if defined(__HCC__)
        getPathV2(path, numSample, dt, state, optionStructs[numOption]);
        #else
        getPathV1(path, numSample, dt, state, optionStructs[numOption]);
        #endif
        dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

        samplePrices[outputNum] = price;
        sampleWeights[outputNum] = DEFAULT_SEQ_WEIGHT;

        //increase the sample and output number if processing multiple samples per thread
        //    numSample += NUM_THREADS_PER_OPTION;
        //    outputNum += NUM_THREADS_PER_OPTION;
    }
}

__global__ void monteCarloGpuKernel(dataType * samplePrices,
                                    dataType * sampleWeights,
                                    dataType * times,
                                    dataType dt,
                                    hiprandStatePhilox4_32_10_t * state,
                                    monteCarloOptionStruct * optionStructs,
                                    int seedVal,
                                    int numSamples)
{
    (void) times;
    //retrieve the thread number
    size_t numThread = blockIdx.x * blockDim.x + threadIdx.x;

    //retrieve the option number
    int numOption = 0;

    //retrieve the number of sample
    int numSample = numThread;

    size_t outputNum = numSample;

    //while (numSample < numSamples)
    if(numSample < numSamples)
    {
        hiprand_init(seedVal, numSample, 0, &(state[numSample]));
        //declare and initialize the path
        dataType path[SEQUENCE_LENGTH];
        initializePath(path);

        getPath(path, numSample, dt, state, optionStructs[numOption]);
        dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

        samplePrices[outputNum] = price;
        sampleWeights[outputNum] = DEFAULT_SEQ_WEIGHT;

        //increase the sample and output number if processing multiple samples per thread
        //    numSample += NUM_THREADS_PER_OPTION;
        //    outputNum += NUM_THREADS_PER_OPTION;
    }
}
