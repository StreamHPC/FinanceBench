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

void getPathCpu(dataType * path,
                size_t sampleNum,
                dataType dt,
                monteCarloOptionStruct optionStruct,
                unsigned int * seedp)
{
    path[0] = getProcessValX0(optionStruct);

    for(size_t i = 1; i < SEQUENCE_LENGTH; ++i)
    {
        dataType t = i * dt;
        dataType randVal = ((dataType)rand_r(seedp)) / ((dataType) RAND_MAX);
        dataType inverseCumRandVal = compInverseNormDist(randVal);
        path[i] = processEvolve(
                      t, path[i - 1], dt, inverseCumRandVal, optionStruct
                  );
    }
}

void monteCarloKernelOpenMP(dataType * samplePrices,
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
			initializePath(path);

			const int optionStructNum = 0;

			getPathCpu(path, numSample, dt, optionStructs[optionStructNum], &my_seed);
			const dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

			samplePrices[numSample] = price;
			sampleWeights[numSample] = DEFAULT_SEQ_WEIGHT;
		}
	}
}

void monteCarloKernelCpu(dataType * samplePrices,
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
        initializePath(path);

        int optionStructNum = 0;

        getPathCpu(path, numSample, dt, optionStructs[optionStructNum], &seed);
        dataType price = getPrice(path[SEQUENCE_LENGTH - 1]);

        samplePrices[numSample] = price;
        sampleWeights[numSample] = DEFAULT_SEQ_WEIGHT;
    }
}
