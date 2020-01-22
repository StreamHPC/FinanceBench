//Scott Grauer-Gray
//May 10, 2012
//Headers for monte carlo function on the

#ifndef MONTE_CARLO_FUNCS_H
#define MONTE_CARLO_FUNCS_H

//needed for constants related to monte carlo
#include "monteCarloConstants.h"
#include "monteCarloStructs.h"
#include <math.h>

#if defined(__NVCC__)
    #define HOST_DEVICE __host__ __device__ inline
#elif defined(__HCC__)
    #if defined(__HIP_DEVICE_COMPILE__)
        #define HOST_DEVICE __device__ inline
    #else
        #define HOST_DEVICE inline
    #endif
#else
    #define HOST_DEVICE inline
#endif

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
HOST_DEVICE dataType compInverseNormDist(dataType x)
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

HOST_DEVICE dataType interestRateCompoundFact(dataType t,
                                              dataType rate)
{
    //assuming "continuous" option
    return exp(rate * t);
}

HOST_DEVICE dataType interestRateDiscountFact(dataType t,
                                              dataType rate)
{
    return 1.0 / interestRateCompoundFact(t, rate);
}

HOST_DEVICE dataType flatForwardDiscountImpl(dataType t,
                                             dataType rate)
{
    return interestRateDiscountFact(t, rate);
}

HOST_DEVICE dataType yieldTermStructDiscount(dataType t,
                                             dataType rate)
{
    return flatForwardDiscountImpl(t, rate);
}

HOST_DEVICE dataType interestRateImpliedRate(dataType compound,
                                             dataType t)
{
    dataType r = log(compound) / t;
    return r;
}

HOST_DEVICE dataType yieldTermStructForwardRate(dataType t1,
                                                dataType t2,
                                                dataType rate)
{
    dataType compound = interestRateDiscountFact(t1, rate) /
                        interestRateDiscountFact(t2, rate);
    return interestRateImpliedRate(compound, t2 - t1);
}

HOST_DEVICE dataType localVoltLocVol(dataType t,
                                     dataType underlyingLevel,
                                     monteCarloOptionStruct optionStruct)
{
    return optionStruct.voltVal;
}

HOST_DEVICE dataType processDiff(dataType t,
                                 dataType x,
                                 monteCarloOptionStruct optionStruct)
{
    return localVoltLocVol(t, x, optionStruct);
}

HOST_DEVICE dataType processDrift(dataType t,
                                  dataType x,
                                  monteCarloOptionStruct optionStruct)
{
    dataType sigma = processDiff(t, x, optionStruct);
    dataType t1 = t + 0.0001;
    return yieldTermStructForwardRate(t, t1, optionStruct.riskVal) -
           yieldTermStructForwardRate(t, t1, optionStruct.divVal) -
           (0.5 * sigma * sigma);
}

HOST_DEVICE dataType discretizationDrift(dataType t0,
                                         dataType x0,
                                         dataType dt,
                                         monteCarloOptionStruct optionStruct)
{
    return processDrift(t0, x0, optionStruct) * dt;
}

HOST_DEVICE dataType discDiff(dataType t0,
                              dataType x0,
                              dataType dt,
                              monteCarloOptionStruct optionStruct)
{
    return processDiff(t0, x0, optionStruct) * sqrt(dt);
}

HOST_DEVICE dataType stdDeviation(dataType t0,
                                  dataType x0,
                                  dataType dt,
                                  monteCarloOptionStruct optionStruct)
{
    return discDiff(t0, x0, dt, optionStruct);
}

HOST_DEVICE dataType apply(dataType x0,
                           dataType dx)
{
    return (x0 * exp(dx));
}

HOST_DEVICE dataType discDrift(dataType t0,
                               dataType x0,
                               dataType dt,
                               monteCarloOptionStruct optionStruct)
{
    return processDrift(t0, x0, optionStruct) * dt;
}

HOST_DEVICE dataType processEvolve(dataType t0,
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
HOST_DEVICE void getSequence(dataType * sequence,
                             dataType sampleNum)
{
    for(unsigned int iInSeq = 0; iInSeq < SEQUENCE_LENGTH; ++iInSeq)
    {
        sequence[iInSeq] = DEFAULT_SEQ_VAL;
    }
}

HOST_DEVICE dataType getProcessValX0(monteCarloOptionStruct optionStruct)
{
    return optionStruct.underlyingVal;
}

HOST_DEVICE dataType getPrice(dataType val)
{
    return mmax(STRIKE_VAL - val, 0.0f) * DISCOUNT_VAL;
}

//initialize the path
HOST_DEVICE void initializePath(dataType * path)
{
    for(unsigned int i = 0; i < SEQUENCE_LENGTH; ++i)
    {
        path[i] = START_PATH_VAL;
    }
}

#endif //MONTE_CARLO_FUNCS_H
