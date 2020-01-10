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

#include "repoKernelsCpu.h"
#ifdef BUILD_CUDA
#include "repoKernels.cuh"
#include <cuda_runtime.h>

#define CUDA_CALL(error)         \
    ASSERT_EQ(static_cast<cudaError_t>(error),cudaSuccess)
#endif

repoDateStruct intializeDate(int d,
                              int m,
                              int y)
{
    repoDateStruct currDate;

    currDate.day = d;
    currDate.month = m;
    currDate.year = y;
    bool leap = isLeapKernel(y);
    int offset = monthOffsetKernel(m,leap);
    currDate.dateSerialNum = d + offset + yearOffsetKernel(y);

    return currDate;
}

void initArgs(inArgsStruct& inArgsHost,
              int numRepos)
{
    for(int numRepo = 0; numRepo < numRepos; ++numRepo)
    {
        repoDateStruct repoSettlementDate = intializeDate(rand() % 28 + 1, 3 - (rand() % 3), 2000);
        repoDateStruct repoDeliveryDate = intializeDate(rand() % 28 + 1, 9 + (rand() % 3), 2000);
        dataType repoRate = 0.05 + ((float)rand() / (float)RAND_MAX - 0.5) * 0.1;
        int repoCompounding = SIMPLE_INTEREST;
        dataType repoCompoundFreq = 1;
        repoDateStruct bondIssueDate = intializeDate(rand() % 28 + 1, rand() % 12 + 1, 1999 - (rand() % 2));
        repoDateStruct bondMaturityDate = intializeDate(rand() % 28 + 1, rand() % 12 + 1, 2001);

        bondStruct bond;
        bond.startDate = bondIssueDate;
        bond.maturityDate = bondMaturityDate;
        bond.rate = 0.08 + ((float)rand() / (float)RAND_MAX - 0.5) * 0.1;

        dataType bondCouponFrequency = 2;
        dataType bondCleanPrice = 89.97693786;

        repoYieldTermStruct bondCurve;
        bondCurve.refDate = repoSettlementDate;
        bondCurve.calDate = repoSettlementDate;
        bondCurve.forward = -0.1f; // dummy rate
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;
        bondCurve.dayCounter = USE_EXACT_DAY;
        bondCurve.refDate = repoSettlementDate;
        bondCurve.calDate = repoSettlementDate;
        bondCurve.compounding = COMPOUNDED_INTEREST;
        bondCurve.frequency = bondCouponFrequency;

        dataType dummyStrike = 91.5745;
        repoYieldTermStruct repoCurve;
        repoCurve.refDate = repoSettlementDate;
        repoCurve.calDate = repoSettlementDate;
        repoCurve.forward = repoRate;
        repoCurve.compounding = repoCompounding;
        repoCurve.frequency = repoCompoundFreq;
        repoCurve.dayCounter = USE_SERIAL_NUMS;

        inArgsHost.discountCurve[numRepo] = bondCurve;
        inArgsHost.repoCurve[numRepo] = repoCurve;
        inArgsHost.settlementDate[numRepo] = repoSettlementDate;
        inArgsHost.deliveryDate[numRepo] = repoDeliveryDate;
        inArgsHost.maturityDate[numRepo] = bondMaturityDate;
        inArgsHost.repoDeliveryDate[numRepo] = repoDeliveryDate;
        inArgsHost.bondCleanPrice[numRepo] = bondCleanPrice;
        inArgsHost.bond[numRepo] = bond;
        inArgsHost.dummyStrike[numRepo] = dummyStrike;
    }
}

TEST(Bonds, OpenMP)
{
    int numRepos = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (repoYieldTermStruct *)malloc(numRepos * sizeof(repoYieldTermStruct));
    inArgsHost.repoCurve = (repoYieldTermStruct *)malloc(numRepos * sizeof(repoYieldTermStruct));
    inArgsHost.settlementDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.deliveryDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.maturityDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.repoDeliveryDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numRepos * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numRepos * sizeof(dataType));

    initArgs(inArgsHost, numRepos);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

    getRepoResultsCpu(inArgsHost, resultsCpu, numRepos);
    getRepoResultsOpenMP(inArgsHost, resultsMp, numRepos);

    for(int i = 0; i < numRepos; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsMp.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountSettlement[i], resultsMp.accruedAmountSettlement[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountDeliveryDate[i], resultsMp.accruedAmountDeliveryDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.forwardSpotIncome[i], resultsMp.forwardSpotIncome[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.underlyingBondFwd[i], resultsMp.underlyingBondFwd[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoNpv[i], resultsMp.repoNpv[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoCleanForwardPrice[i], resultsMp.repoCleanForwardPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoDirtyForwardPrice[i], resultsMp.repoDirtyForwardPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoImpliedYield[i], resultsMp.repoImpliedYield[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.marketRepoRate[i], resultsMp.marketRepoRate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsMp.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    free(resultsCpu.dirtyPrice);
    free(resultsCpu.accruedAmountSettlement);
    free(resultsCpu.accruedAmountDeliveryDate);
    free(resultsCpu.cleanPrice);
    free(resultsCpu.forwardSpotIncome);
    free(resultsCpu.underlyingBondFwd);
    free(resultsCpu.repoNpv);
    free(resultsCpu.repoCleanForwardPrice);
    free(resultsCpu.repoDirtyForwardPrice);
    free(resultsCpu.repoImpliedYield);
    free(resultsCpu.marketRepoRate);
    free(resultsCpu.bondForwardVal);

    free(resultsMp.dirtyPrice);
    free(resultsMp.accruedAmountSettlement);
    free(resultsMp.accruedAmountDeliveryDate);
    free(resultsMp.cleanPrice);
    free(resultsMp.forwardSpotIncome);
    free(resultsMp.underlyingBondFwd);
    free(resultsMp.repoNpv);
    free(resultsMp.repoCleanForwardPrice);
    free(resultsMp.repoDirtyForwardPrice);
    free(resultsMp.repoImpliedYield);
    free(resultsMp.marketRepoRate);
    free(resultsMp.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.settlementDate);
    free(inArgsHost.deliveryDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.repoDeliveryDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}

#ifdef BUILD_CUDA
TEST(Bonds, Cuda)
{
    int numRepos = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (repoYieldTermStruct *)malloc(numRepos * sizeof(repoYieldTermStruct));
    inArgsHost.repoCurve = (repoYieldTermStruct *)malloc(numRepos * sizeof(repoYieldTermStruct));
    inArgsHost.settlementDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.deliveryDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.maturityDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.repoDeliveryDate = (repoDateStruct *)malloc(numRepos * sizeof(repoDateStruct));
    inArgsHost.bondCleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    inArgsHost.bond = (bondStruct *)malloc(numRepos * sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType *)malloc(numRepos * sizeof(dataType));

    initArgs(inArgsHost, numRepos);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsCpu.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsMp.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

    getRepoResultsCpu(inArgsHost, resultsCpu, numRepos);

    inArgsStruct inArgsGpu;
    resultsStruct resultsGpu;

    CUDA_CALL(cudaMalloc(&(resultsGpu.dirtyPrice), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.cleanPrice), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.repoNpv), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.repoImpliedYield), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.marketRepoRate), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(resultsGpu.bondForwardVal), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.discountCurve), numRepos * sizeof(repoYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.repoCurve), numRepos * sizeof(repoYieldTermStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.settlementDate), numRepos * sizeof(repoDateStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.deliveryDate), numRepos * sizeof(repoDateStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.maturityDate), numRepos * sizeof(repoDateStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.repoDeliveryDate), numRepos * sizeof(repoDateStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.bondCleanPrice), numRepos * sizeof(dataType)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.bond), numRepos * sizeof(bondStruct)));
    CUDA_CALL(cudaMalloc(&(inArgsGpu.dummyStrike), numRepos * sizeof(dataType)));

    CUDA_CALL(cudaMemcpy((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), cudaMemcpyHostToDevice));

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    getRepoResultsGpu<<<gridDim, blockDim>>>(inArgsGpu, resultsGpu, numRepos);
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaMemcpy(resultsMp.dirtyPrice, (resultsGpu.dirtyPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.accruedAmountSettlement, (resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.accruedAmountDeliveryDate, (resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.cleanPrice, (resultsGpu.cleanPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.forwardSpotIncome, (resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.underlyingBondFwd, (resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.repoNpv, (resultsGpu.repoNpv), numRepos *sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.repoCleanForwardPrice, (resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.repoDirtyForwardPrice, (resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.repoImpliedYield, (resultsGpu.repoImpliedYield), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.marketRepoRate, (resultsGpu.marketRepoRate), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(resultsMp.bondForwardVal, (resultsGpu.bondForwardVal), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for(int i = 0; i < numRepos; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsMp.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountSettlement[i], resultsMp.accruedAmountSettlement[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountDeliveryDate[i], resultsMp.accruedAmountDeliveryDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.forwardSpotIncome[i], resultsMp.forwardSpotIncome[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.underlyingBondFwd[i], resultsMp.underlyingBondFwd[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoNpv[i], resultsMp.repoNpv[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoCleanForwardPrice[i], resultsMp.repoCleanForwardPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoDirtyForwardPrice[i], resultsMp.repoDirtyForwardPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.repoImpliedYield[i], resultsMp.repoImpliedYield[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.marketRepoRate[i], resultsMp.marketRepoRate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsMp.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    CUDA_CALL(cudaFree(resultsGpu.dirtyPrice));
    CUDA_CALL(cudaFree(resultsGpu.accruedAmountSettlement));
    CUDA_CALL(cudaFree(resultsGpu.accruedAmountDeliveryDate));
    CUDA_CALL(cudaFree(resultsGpu.cleanPrice));
    CUDA_CALL(cudaFree(resultsGpu.forwardSpotIncome));
    CUDA_CALL(cudaFree(resultsGpu.underlyingBondFwd));
    CUDA_CALL(cudaFree(resultsGpu.repoNpv));
    CUDA_CALL(cudaFree(resultsGpu.repoCleanForwardPrice));
    CUDA_CALL(cudaFree(resultsGpu.repoDirtyForwardPrice));
    CUDA_CALL(cudaFree(resultsGpu.repoImpliedYield));
    CUDA_CALL(cudaFree(resultsGpu.marketRepoRate));
    CUDA_CALL(cudaFree(resultsGpu.bondForwardVal));
    CUDA_CALL(cudaFree(inArgsGpu.discountCurve));
    CUDA_CALL(cudaFree(inArgsGpu.repoCurve));
    CUDA_CALL(cudaFree(inArgsGpu.settlementDate));
    CUDA_CALL(cudaFree(inArgsGpu.deliveryDate));
    CUDA_CALL(cudaFree(inArgsGpu.maturityDate));
    CUDA_CALL(cudaFree(inArgsGpu.repoDeliveryDate));
    CUDA_CALL(cudaFree(inArgsGpu.bondCleanPrice));
    CUDA_CALL(cudaFree(inArgsGpu.bond));
    CUDA_CALL(cudaFree(inArgsGpu.dummyStrike));

    free(resultsCpu.dirtyPrice);
    free(resultsCpu.accruedAmountSettlement);
    free(resultsCpu.accruedAmountDeliveryDate);
    free(resultsCpu.cleanPrice);
    free(resultsCpu.forwardSpotIncome);
    free(resultsCpu.underlyingBondFwd);
    free(resultsCpu.repoNpv);
    free(resultsCpu.repoCleanForwardPrice);
    free(resultsCpu.repoDirtyForwardPrice);
    free(resultsCpu.repoImpliedYield);
    free(resultsCpu.marketRepoRate);
    free(resultsCpu.bondForwardVal);

    free(resultsMp.dirtyPrice);
    free(resultsMp.accruedAmountSettlement);
    free(resultsMp.accruedAmountDeliveryDate);
    free(resultsMp.cleanPrice);
    free(resultsMp.forwardSpotIncome);
    free(resultsMp.underlyingBondFwd);
    free(resultsMp.repoNpv);
    free(resultsMp.repoCleanForwardPrice);
    free(resultsMp.repoDirtyForwardPrice);
    free(resultsMp.repoImpliedYield);
    free(resultsMp.marketRepoRate);
    free(resultsMp.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.settlementDate);
    free(inArgsHost.deliveryDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.repoDeliveryDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
}
#endif
