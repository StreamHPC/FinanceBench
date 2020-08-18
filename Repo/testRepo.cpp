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
    inArgsHost.discountCurve = new repoYieldTermStruct[numRepos];
    inArgsHost.repoCurve = new repoYieldTermStruct[numRepos];
    inArgsHost.settlementDate = new repoDateStruct[numRepos];
    inArgsHost.deliveryDate = new repoDateStruct[numRepos];
    inArgsHost.maturityDate = new repoDateStruct[numRepos];
    inArgsHost.repoDeliveryDate = new repoDateStruct[numRepos];
    inArgsHost.bondCleanPrice = new dataType[numRepos];
    inArgsHost.bond = new bondStruct[numRepos];
    inArgsHost.dummyStrike = new dataType[numRepos];

    initArgs(inArgsHost, numRepos);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = new dataType[numRepos];
    resultsCpu.accruedAmountSettlement = new dataType[numRepos];
    resultsCpu.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsCpu.cleanPrice = new dataType[numRepos];
    resultsCpu.forwardSpotIncome = new dataType[numRepos];
    resultsCpu.underlyingBondFwd = new dataType[numRepos];
    resultsCpu.repoNpv = new dataType[numRepos];
    resultsCpu.repoCleanForwardPrice = new dataType[numRepos];
    resultsCpu.repoDirtyForwardPrice = new dataType[numRepos];
    resultsCpu.repoImpliedYield = new dataType[numRepos];
    resultsCpu.marketRepoRate = new dataType[numRepos];
    resultsCpu.bondForwardVal = new dataType[numRepos];

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = new dataType[numRepos];
    resultsMp.accruedAmountSettlement = new dataType[numRepos];
    resultsMp.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsMp.cleanPrice = new dataType[numRepos];
    resultsMp.forwardSpotIncome = new dataType[numRepos];
    resultsMp.underlyingBondFwd = new dataType[numRepos];
    resultsMp.repoNpv = new dataType[numRepos];
    resultsMp.repoCleanForwardPrice = new dataType[numRepos];
    resultsMp.repoDirtyForwardPrice = new dataType[numRepos];
    resultsMp.repoImpliedYield = new dataType[numRepos];
    resultsMp.marketRepoRate = new dataType[numRepos];
    resultsMp.bondForwardVal = new dataType[numRepos];

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

    delete [] resultsCpu.dirtyPrice;
    delete [] resultsCpu.accruedAmountSettlement;
    delete [] resultsCpu.accruedAmountDeliveryDate;
    delete [] resultsCpu.cleanPrice;
    delete [] resultsCpu.forwardSpotIncome;
    delete [] resultsCpu.underlyingBondFwd;
    delete [] resultsCpu.repoNpv;
    delete [] resultsCpu.repoCleanForwardPrice;
    delete [] resultsCpu.repoDirtyForwardPrice;
    delete [] resultsCpu.repoImpliedYield;
    delete [] resultsCpu.marketRepoRate;
    delete [] resultsCpu.bondForwardVal;

    delete [] resultsMp.dirtyPrice;
    delete [] resultsMp.accruedAmountSettlement;
    delete [] resultsMp.accruedAmountDeliveryDate;
    delete [] resultsMp.cleanPrice;
    delete [] resultsMp.forwardSpotIncome;
    delete [] resultsMp.underlyingBondFwd;
    delete [] resultsMp.repoNpv;
    delete [] resultsMp.repoCleanForwardPrice;
    delete [] resultsMp.repoDirtyForwardPrice;
    delete [] resultsMp.repoImpliedYield;
    delete [] resultsMp.marketRepoRate;
    delete [] resultsMp.bondForwardVal;

    delete [] inArgsHost.discountCurve;
    delete [] inArgsHost.repoCurve;
    delete [] inArgsHost.settlementDate;
    delete [] inArgsHost.deliveryDate;
    delete [] inArgsHost.maturityDate;
    delete [] inArgsHost.repoDeliveryDate;
    delete [] inArgsHost.bondCleanPrice;
    delete [] inArgsHost.bond;
    delete [] inArgsHost.dummyStrike;
}

#ifdef BUILD_HIP
TEST(Bonds, Hip)
{
    int numRepos = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = new repoYieldTermStruct[numRepos];
    inArgsHost.repoCurve = new repoYieldTermStruct[numRepos];
    inArgsHost.settlementDate = new repoDateStruct[numRepos];
    inArgsHost.deliveryDate = new repoDateStruct[numRepos];
    inArgsHost.maturityDate = new repoDateStruct[numRepos];
    inArgsHost.repoDeliveryDate = new repoDateStruct[numRepos];
    inArgsHost.bondCleanPrice = new dataType[numRepos];
    inArgsHost.bond = new bondStruct[numRepos];
    inArgsHost.dummyStrike = new dataType[numRepos];

    initArgs(inArgsHost, numRepos);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = new dataType[numRepos];
    resultsCpu.accruedAmountSettlement = new dataType[numRepos];
    resultsCpu.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsCpu.cleanPrice = new dataType[numRepos];
    resultsCpu.forwardSpotIncome = new dataType[numRepos];
    resultsCpu.underlyingBondFwd = new dataType[numRepos];
    resultsCpu.repoNpv = new dataType[numRepos];
    resultsCpu.repoCleanForwardPrice = new dataType[numRepos];
    resultsCpu.repoDirtyForwardPrice = new dataType[numRepos];
    resultsCpu.repoImpliedYield = new dataType[numRepos];
    resultsCpu.marketRepoRate = new dataType[numRepos];
    resultsCpu.bondForwardVal = new dataType[numRepos];

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = new dataType[numRepos];
    resultsMp.accruedAmountSettlement = new dataType[numRepos];
    resultsMp.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsMp.cleanPrice = new dataType[numRepos];
    resultsMp.forwardSpotIncome = new dataType[numRepos];
    resultsMp.underlyingBondFwd = new dataType[numRepos];
    resultsMp.repoNpv = new dataType[numRepos];
    resultsMp.repoCleanForwardPrice = new dataType[numRepos];
    resultsMp.repoDirtyForwardPrice = new dataType[numRepos];
    resultsMp.repoImpliedYield = new dataType[numRepos];
    resultsMp.marketRepoRate = new dataType[numRepos];
    resultsMp.bondForwardVal = new dataType[numRepos];

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

    delete [] resultsCpu.dirtyPrice;
    delete [] resultsCpu.accruedAmountSettlement;
    delete [] resultsCpu.accruedAmountDeliveryDate;
    delete [] resultsCpu.cleanPrice;
    delete [] resultsCpu.forwardSpotIncome;
    delete [] resultsCpu.underlyingBondFwd;
    delete [] resultsCpu.repoNpv;
    delete [] resultsCpu.repoCleanForwardPrice;
    delete [] resultsCpu.repoDirtyForwardPrice;
    delete [] resultsCpu.repoImpliedYield;
    delete [] resultsCpu.marketRepoRate;
    delete [] resultsCpu.bondForwardVal;

    delete [] resultsMp.dirtyPrice;
    delete [] resultsMp.accruedAmountSettlement;
    delete [] resultsMp.accruedAmountDeliveryDate;
    delete [] resultsMp.cleanPrice;
    delete [] resultsMp.forwardSpotIncome;
    delete [] resultsMp.underlyingBondFwd;
    delete [] resultsMp.repoNpv;
    delete [] resultsMp.repoCleanForwardPrice;
    delete [] resultsMp.repoDirtyForwardPrice;
    delete [] resultsMp.repoImpliedYield;
    delete [] resultsMp.marketRepoRate;
    delete [] resultsMp.bondForwardVal;

    delete [] inArgsHost.discountCurve;
    delete [] inArgsHost.repoCurve;
    delete [] inArgsHost.settlementDate;
    delete [] inArgsHost.deliveryDate;
    delete [] inArgsHost.maturityDate;
    delete [] inArgsHost.repoDeliveryDate;
    delete [] inArgsHost.bondCleanPrice;
    delete [] inArgsHost.bond;
    delete [] inArgsHost.dummyStrike;
}
#endif
