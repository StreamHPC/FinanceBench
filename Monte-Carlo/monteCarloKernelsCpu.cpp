//monteCarloKernelsCpu.cu
//Scott Grauer-Gray
//May 10, 2012
//monte carlo kernels run on the CPU

#include "monteCarloKernelsCpu.h"

//function to set up the random states
void setup_kernelCpu()
{
    srand(time(NULL));
}

//function to compute the inverse normal distribution
dataType compInverseNormDistCpu(dataType x)
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

dataType interestRateCompoundFactCpu(dataType t,
                                  dataType rate)
{
    //assuming "continuous" option
    return exp(rate * t);
}

dataType interestRateDiscountFactCpu(dataType t,
                                  dataType rate)
{
    return 1.0 / interestRateCompoundFactCpu(t, rate);
}

dataType flatForwardDiscountImplCpu(dataType t,
                                 dataType rate)
{
    return interestRateDiscountFactCpu(t, rate);
}

dataType yieldTermStructDiscountCpu(dataType t,
                                 dataType rate)
{
    return flatForwardDiscountImplCpu(t, rate);
}

dataType interestRateImpliedRateCpu(dataType compound,
                                 dataType t)
{
    dataType r = log(compound) / t;
    return r;
}

dataType yieldTermStructForwardRateCpu(dataType t1,
                                    dataType t2,
                                    dataType rate)
{
    dataType compound = interestRateDiscountFactCpu(t1, rate) /
                        interestRateDiscountFactCpu(t2, rate);
    return interestRateImpliedRateCpu(compound, t2 - t1);
}

dataType processDriftCpu(dataType t,
                      dataType x,
                      monteCarloOptionStruct optionStruct)
{
    dataType sigma = processDiffCpu(t, x, optionStruct);
    dataType t1 = t + 0.0001;
    return yieldTermStructForwardRateCpu(t, t1, optionStruct.riskVal) -
           yieldTermStructForwardRateCpu(t, t1, optionStruct.divVal) -
           (0.5 * sigma * sigma);
}

dataType discretizationDriftCpu(dataType t0,
                             dataType x0,
                             dataType dt,
                             monteCarloOptionStruct optionStruct)
{
    return processDriftCpu(t0, x0, optionStruct) * dt;
}

dataType localVoltLocVolCpu(dataType t,
                         dataType underlyingLevel,
                         monteCarloOptionStruct optionStruct)
{
    return optionStruct.voltVal;
}

dataType processDiffCpu(dataType t,
                     dataType x,
                     monteCarloOptionStruct optionStruct)
{
    return localVoltLocVolCpu(t, x, optionStruct);
}

dataType discDiffCpu(dataType t0,
                  dataType x0,
                  dataType dt,
                  monteCarloOptionStruct optionStruct)
{
    return processDiffCpu(t0, x0, optionStruct) * sqrt(dt);
}

dataType stdDeviationCpu(dataType t0,
                      dataType x0,
                      dataType dt,
                      monteCarloOptionStruct optionStruct)
{
    return discDiffCpu(t0, x0, dt, optionStruct);
}

dataType applyCpu(dataType x0,
               dataType dx)
{
    return (x0 * exp(dx));
}

dataType discDriftCpu(dataType t0,
                   dataType x0,
                   dataType dt,
                   monteCarloOptionStruct optionStruct)
{
    return processDriftCpu(t0, x0, optionStruct) * dt;
}

dataType processEvolveCpu(dataType t0,
                       dataType x0,
                       dataType dt,
                       dataType dw,
                       monteCarloOptionStruct optionStruct)
{
    return applyCpu(
        x0, discDriftCpu(t0, x0, dt, optionStruct) +
        stdDeviationCpu(t0, x0, dt, optionStruct) * dw
    );
}

//retrieve the current sequence
void getSequenceCpu(dataType * sequence,
                    dataType sampleNum)
{
    for(size_t iInSeq = 0; iInSeq < SEQUENCE_LENGTH; ++iInSeq)
    {
        sequence[iInSeq] = DEFAULT_SEQ_VAL;
    }
}

dataType getProcessValX0Cpu(monteCarloOptionStruct optionStruct)
{
    return optionStruct.underlyingVal;
}

void getPathCpu(dataType * path,
                size_t sampleNum,
                dataType dt,
                monteCarloOptionStruct optionStruct,
                unsigned int * seedp)
{
    path[0] = getProcessValX0Cpu(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        dataType randVal = ((dataType)rand_r(seedp)) / ((dataType) RAND_MAX);
        dataType inverseCumRandVal = compInverseNormDistCpu(randVal);
        path[i] = processEvolveCpu(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

dataType getPriceCpu(dataType val)
{
    return fmax(STRIKE_VAL - val, 0.0) * DISCOUNT_VAL;
}

//initialize the path
void initializePathCpu(dataType * path)
{
    for(int i = 0; i < SEQUENCE_LENGTH; ++i)
    {
        path[i] = START_PATH_VAL;
    }
}

void monteCarloGpuKernelOpenMP(dataType * samplePrices,
                               dataType * sampleWeights,
                               dataType * times,
                               dataType dt,
                               monteCarloOptionStruct * optionStructs,
                               unsigned int seed,
                               int numSamples)
{
    #pragma omp parallel
	{
		unsigned int my_id = omp_get_thread_num();
		unsigned int my_seed = seed + my_id;
		#pragma omp for schedule(static, 1000)
		for(size_t numSample = 0; numSample < numSamples; ++numSample)
		{
			// Declare and initialize the path.
			dataType path[SEQUENCE_LENGTH];
			initializePathCpu(path);

			const int optionStructNum = 0;

			getPathCpu(path, numSample, dt, optionStructs[optionStructNum], &my_seed);
			const dataType price = getPriceCpu(path[SEQUENCE_LENGTH - 1]);

			samplePrices[numSample] = price;
			sampleWeights[numSample] = DEFAULT_SEQ_WEIGHT;
		}
	}
}

void monteCarloGpuKernelCpu(dataType * samplePrices,
                            dataType * sampleWeights,
                            dataType * times,
                            dataType dt,
                            monteCarloOptionStruct * optionStructs,
                            unsigned int seed,
                            int numSamples)
{
    for(size_t numSample = 0; numSample < numSamples; ++numSample)
    {
        //declare and initialize the path
        dataType path[SEQUENCE_LENGTH];
        initializePathCpu(path);

        int optionStructNum = 0;

        getPathCpu(path, numSample, dt, optionStructs[optionStructNum], &seed);
        dataType price = getPriceCpu(path[SEQUENCE_LENGTH - 1]);

        samplePrices[numSample] = price;
        sampleWeights[numSample] = DEFAULT_SEQ_WEIGHT;
    }
}
