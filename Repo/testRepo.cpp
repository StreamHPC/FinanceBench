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
#ifdef BUILD_HIP
#include "repoKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)
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

#ifdef BUILD_HIP
TEST(Bonds, Hip)
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

    HIP_CALL(hipMalloc(&(resultsGpu.dirtyPrice), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.cleanPrice), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.repoNpv), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.repoImpliedYield), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.marketRepoRate), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(resultsGpu.bondForwardVal), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(inArgsGpu.discountCurve), numRepos * sizeof(repoYieldTermStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.repoCurve), numRepos * sizeof(repoYieldTermStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.settlementDate), numRepos * sizeof(repoDateStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.deliveryDate), numRepos * sizeof(repoDateStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.maturityDate), numRepos * sizeof(repoDateStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.repoDeliveryDate), numRepos * sizeof(repoDateStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.bondCleanPrice), numRepos * sizeof(dataType)));
    HIP_CALL(hipMalloc(&(inArgsGpu.bond), numRepos * sizeof(bondStruct)));
    HIP_CALL(hipMalloc(&(inArgsGpu.dummyStrike), numRepos * sizeof(dataType)));

    HIP_CALL(hipMemcpy((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), hipMemcpyHostToDevice));

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    hipLaunchKernelGGL((getRepoResultsGpu), dim3(gridDim), dim3(blockDim), 0, 0, inArgsGpu, resultsGpu, numRepos);
    HIP_CALL(hipPeekAtLastError());

    HIP_CALL(hipMemcpy(resultsMp.dirtyPrice, (resultsGpu.dirtyPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.accruedAmountSettlement, (resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.accruedAmountDeliveryDate, (resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType),hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.cleanPrice, (resultsGpu.cleanPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.forwardSpotIncome, (resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.underlyingBondFwd, (resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.repoNpv, (resultsGpu.repoNpv), numRepos *sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.repoCleanForwardPrice, (resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.repoDirtyForwardPrice, (resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.repoImpliedYield, (resultsGpu.repoImpliedYield), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.marketRepoRate, (resultsGpu.marketRepoRate), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsMp.bondForwardVal, (resultsGpu.bondForwardVal), numRepos * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

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

    HIP_CALL(hipFree(resultsGpu.dirtyPrice));
    HIP_CALL(hipFree(resultsGpu.accruedAmountSettlement));
    HIP_CALL(hipFree(resultsGpu.accruedAmountDeliveryDate));
    HIP_CALL(hipFree(resultsGpu.cleanPrice));
    HIP_CALL(hipFree(resultsGpu.forwardSpotIncome));
    HIP_CALL(hipFree(resultsGpu.underlyingBondFwd));
    HIP_CALL(hipFree(resultsGpu.repoNpv));
    HIP_CALL(hipFree(resultsGpu.repoCleanForwardPrice));
    HIP_CALL(hipFree(resultsGpu.repoDirtyForwardPrice));
    HIP_CALL(hipFree(resultsGpu.repoImpliedYield));
    HIP_CALL(hipFree(resultsGpu.marketRepoRate));
    HIP_CALL(hipFree(resultsGpu.bondForwardVal));
    HIP_CALL(hipFree(inArgsGpu.discountCurve));
    HIP_CALL(hipFree(inArgsGpu.repoCurve));
    HIP_CALL(hipFree(inArgsGpu.settlementDate));
    HIP_CALL(hipFree(inArgsGpu.deliveryDate));
    HIP_CALL(hipFree(inArgsGpu.maturityDate));
    HIP_CALL(hipFree(inArgsGpu.repoDeliveryDate));
    HIP_CALL(hipFree(inArgsGpu.bondCleanPrice));
    HIP_CALL(hipFree(inArgsGpu.bond));
    HIP_CALL(hipFree(inArgsGpu.dummyStrike));

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
