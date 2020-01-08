//monteCarloKernels.cu
//Scott Grauer-Gray
//May 10, 2012
//GPU Kernels for running monte carlo

#include "monteCarloKernels.cuh"

//function to set up the random states
__global__ void setup_kernel(curandState * state,
                             int seedVal,
                             int numSamples)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numSamples)
    {
        /* Each thread gets same seed , a different sequence
        number , no offset */
        curand_init(seedVal, id, 0, &(state[id]));
    }
}
