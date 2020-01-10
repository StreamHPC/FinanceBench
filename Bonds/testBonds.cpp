#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>

#include "bondsKernelsCpu.h"
#ifdef BUILD_CUDA
#include "bondsKernelsGpu.cuh"
#include <cuda_runtime.h>

#define CUDA_CALL(error)         \
    ASSERT_EQ(static_cast<cudaError_t>(error),cudaSuccess)
#endif

bondsDateStruct intializeDate(int d,
                              int m,
                              int y)
{
	bondsDateStruct currDate;

	currDate.day = d;
	currDate.month = m;
	currDate.year = y;
	bool leap = isLeapKernel(y);
	int offset = monthOffsetKernel(m,leap);
	currDate.dateSerialNum = d + offset + yearOffsetKernel(y);

	return currDate;
}

void initArgs(inArgsStruct& inArgsHost,
              int numBonds)
{
    int numBond;
    for(numBond = 0; numBond < numBonds; ++numBond)
    {
        dataType repoRate = 0.07;
        int repoCompounding = SIMPLE_INTEREST;
        dataType repoCompoundFreq = 1;

        bondsDateStruct bondIssueDate =  intializeDate(rand() % 28 + 1, rand() % 12 + 1, 1999 - (rand() % 2));
        bondsDateStruct bondMaturityDate = intializeDate(rand() % 28 + 1, rand() % 12 + 1, 2000 + (rand() % 2));
        bondsDateStruct todaysDate = intializeDate(bondMaturityDate.day-1,bondMaturityDate.month,bondMaturityDate.year);

        bondStruct bond;
        bond.startDate = bondIssueDate;
        bond.maturityDate = bondMaturityDate;
        bond.rate = 0.08 + ((float)rand()/(float)RAND_MAX - 0.5)*0.1;

        dataType bondCouponFrequency = 2;
        dataType bondCleanPrice = 89.97693786;

        bondsYieldTermStruct bondCurve;
        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.forward = -0.1f;  // dummy rate
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;
        bondCurve.dayCounter = USE_EXACT_DAY;
        bondCurve.refDate = todaysDate;
        bondCurve.calDate = todaysDate;
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;

        dataType dummyStrike = 91.5745;
        bondsYieldTermStruct repoCurve;
        repoCurve.refDate = todaysDate;
        repoCurve.calDate = todaysDate;
        repoCurve.forward = repoRate;
        repoCurve.compounding = repoCompounding;
        repoCurve.frequency = repoCompoundFreq;
        repoCurve.dayCounter = USE_SERIAL_NUMS;
        inArgsHost.discountCurve[numBond] = bondCurve;
        inArgsHost.repoCurve[numBond] = repoCurve;
        inArgsHost.currDate[numBond] = todaysDate;
        inArgsHost.maturityDate[numBond] = bondMaturityDate;
        inArgsHost.bondCleanPrice[numBond] = bondCleanPrice;
        inArgsHost.bond[numBond] = bond;
        inArgsHost.dummyStrike[numBond] = dummyStrike;
    }
}

TEST(Bonds, OpenMP)
{
    int numBonds = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsMp.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsMp.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsMp.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    getBondsResultsCpu(inArgsHost, resultsCpu, numBonds);
    getBondsResultsOpenMP(inArgsHost, resultsMp, numBonds);

    for(int i = 0; i < numBonds; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsMp.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountCurrDate[i], resultsMp.accruedAmountCurrDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsMp.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    free(resultsCpu.dirtyPrice);
    free(resultsCpu.accruedAmountCurrDate);
    free(resultsCpu.cleanPrice);
    free(resultsCpu.bondForwardVal);
    free(resultsMp.dirtyPrice);
    free(resultsMp.accruedAmountCurrDate);
    free(resultsMp.cleanPrice);
    free(resultsMp.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

#ifdef BUILD_CUDA
TEST(Bonds, Cuda)
{
    int numBonds = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct *)malloc(numBonds * sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct *)malloc(numBonds * sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numBonds * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numBonds * sizeof(dataType));

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsCpu.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    resultsStruct resultsGpu;
    resultsGpu.dirtyPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsGpu.accruedAmountCurrDate = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsGpu.cleanPrice = (dataType *)malloc(numBonds * sizeof(dataType));
    resultsGpu.bondForwardVal = (dataType *)malloc(numBonds * sizeof(dataType));

    getBondsResultsCpu(inArgsHost, resultsCpu, numBonds);

    bondsYieldTermStruct * discountCurveGpu;
    bondsYieldTermStruct * repoCurveGpu;
    bondsDateStruct * currDateGpu;
    bondsDateStruct * maturityDateGpu;
    dataType * bondCleanPriceGpu;
    bondStruct * bondGpu;
    dataType * dummyStrikeGpu;
    dataType * dirtyPriceGpu;
    dataType * accruedAmountCurrDateGpu;
    dataType * cleanPriceGpu;
    dataType * bondForwardValGpu;

    dim3 grid((ceil(((float)numBonds)/((float)256))), 1, 1);
    dim3 threads(256, 1, 1);

    CUDA_CALL(cudaMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    CUDA_CALL(cudaMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    CUDA_CALL(cudaMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

    CUDA_CALL(cudaMemcpy(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), cudaMemcpyHostToDevice));

    inArgsStruct inArgs;
    inArgs.discountCurve    = discountCurveGpu;
    inArgs.repoCurve        = repoCurveGpu;
    inArgs.currDate   = currDateGpu;
    inArgs.maturityDate     = maturityDateGpu;
    inArgs.bondCleanPrice   = bondCleanPriceGpu;
    inArgs.bond             = bondGpu;
    inArgs.dummyStrike      = dummyStrikeGpu;

    resultsStruct results;
    results.dirtyPrice                = dirtyPriceGpu;
    results.accruedAmountCurrDate  = accruedAmountCurrDateGpu;
    results.cleanPrice                = cleanPriceGpu;
    results.bondForwardVal         = bondForwardValGpu;

    getBondsResultsGpu<<<grid, threads>>>(inArgs, results, numBonds);
    CUDA_CALL(cudaMemcpy(resultsGpu.dirtyPrice, dirtyPriceGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsGpu.accruedAmountCurrDate, accruedAmountCurrDateGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsGpu.cleanPrice, cleanPriceGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsGpu.bondForwardVal, bondForwardValGpu, numBonds * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for(int i = 0; i < numBonds; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsGpu.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountCurrDate[i], resultsGpu.accruedAmountCurrDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsGpu.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsGpu.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    CUDA_CALL(cudaFree(discountCurveGpu));
    CUDA_CALL(cudaFree(repoCurveGpu));
    CUDA_CALL(cudaFree(currDateGpu));
    CUDA_CALL(cudaFree(maturityDateGpu));
    CUDA_CALL(cudaFree(bondCleanPriceGpu));
    CUDA_CALL(cudaFree(bondGpu));
    CUDA_CALL(cudaFree(dummyStrikeGpu));
    CUDA_CALL(cudaFree(dirtyPriceGpu));
    CUDA_CALL(cudaFree(accruedAmountCurrDateGpu));
    CUDA_CALL(cudaFree(cleanPriceGpu));
    CUDA_CALL(cudaFree(bondForwardValGpu));

    free(resultsCpu.dirtyPrice);
    free(resultsCpu.accruedAmountCurrDate);
    free(resultsCpu.cleanPrice);
    free(resultsCpu.bondForwardVal);
    free(resultsGpu.dirtyPrice);
    free(resultsGpu.accruedAmountCurrDate);
    free(resultsGpu.cleanPrice);
    free(resultsGpu.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}
#endif
