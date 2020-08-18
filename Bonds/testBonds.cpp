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
#ifdef BUILD_HIP
#include "bondsKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)
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
    inArgsHost.discountCurve = new bondsYieldTermStruct[numBonds];
    inArgsHost.repoCurve = new bondsYieldTermStruct[numBonds];
    inArgsHost.currDate = new bondsDateStruct[numBonds];
    inArgsHost.maturityDate = new bondsDateStruct[numBonds];
    inArgsHost.bondCleanPrice = new dataType[numBonds];
    inArgsHost.bond = new bondStruct[numBonds];
    inArgsHost.dummyStrike = new dataType[numBonds];

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = new dataType[numBonds];
    resultsCpu.accruedAmountCurrDate = new dataType[numBonds];
    resultsCpu.cleanPrice = new dataType[numBonds];
    resultsCpu.bondForwardVal = new dataType[numBonds];

    resultsStruct resultsMp;
    resultsMp.dirtyPrice = new dataType[numBonds];
    resultsMp.accruedAmountCurrDate = new dataType[numBonds];
    resultsMp.cleanPrice = new dataType[numBonds];
    resultsMp.bondForwardVal = new dataType[numBonds];

    getBondsResultsCpu(inArgsHost, resultsCpu, numBonds);
    getBondsResultsOpenMP(inArgsHost, resultsMp, numBonds);

    for(int i = 0; i < numBonds; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsMp.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountCurrDate[i], resultsMp.accruedAmountCurrDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsMp.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsMp.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    delete [] resultsCpu.dirtyPrice;
    delete [] resultsCpu.accruedAmountCurrDate;
    delete [] resultsCpu.cleanPrice;
    delete [] resultsCpu.bondForwardVal;
    delete [] resultsMp.dirtyPrice;
    delete [] resultsMp.accruedAmountCurrDate;
    delete [] resultsMp.cleanPrice;
    delete [] resultsMp.bondForwardVal;

    delete [] inArgsHost.discountCurve;
    delete [] inArgsHost.repoCurve;
    delete [] inArgsHost.currDate;
    delete [] inArgsHost.maturityDate;
    delete [] inArgsHost.bondCleanPrice;
    delete [] inArgsHost.bond;
    delete [] inArgsHost.dummyStrike;
}

#ifdef BUILD_HIP
TEST(Bonds, Hip)
{
    int numBonds = 1024;
    const int seed = 123;
    srand(seed);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = new bondsYieldTermStruct[numBonds];
    inArgsHost.repoCurve = new bondsYieldTermStruct[numBonds];
    inArgsHost.currDate = new bondsDateStruct[numBonds];
    inArgsHost.maturityDate = new bondsDateStruct[numBonds];
    inArgsHost.bondCleanPrice = new dataType[numBonds];
    inArgsHost.bond = new bondStruct[numBonds];
    inArgsHost.dummyStrike = new dataType[numBonds];

    initArgs(inArgsHost, numBonds);

    resultsStruct resultsCpu;
    resultsCpu.dirtyPrice = new dataType[numBonds];
    resultsCpu.accruedAmountCurrDate = new dataType[numBonds];
    resultsCpu.cleanPrice = new dataType[numBonds];
    resultsCpu.bondForwardVal = new dataType[numBonds];

    resultsStruct resultsGpu;
    resultsGpu.dirtyPrice = new dataType[numBonds];
    resultsGpu.accruedAmountCurrDate = new dataType[numBonds];
    resultsGpu.cleanPrice = new dataType[numBonds];
    resultsGpu.bondForwardVal = new dataType[numBonds];

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

    HIP_CALL(hipMalloc(&discountCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&repoCurveGpu, numBonds * sizeof(bondsYieldTermStruct)));
    HIP_CALL(hipMalloc(&currDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&maturityDateGpu, numBonds * sizeof(bondsDateStruct)));
    HIP_CALL(hipMalloc(&bondCleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondGpu, numBonds * sizeof(bondStruct)));
    HIP_CALL(hipMalloc(&dummyStrikeGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&dirtyPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&accruedAmountCurrDateGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&cleanPriceGpu, numBonds * sizeof(dataType)));
    HIP_CALL(hipMalloc(&bondForwardValGpu, numBonds * sizeof(dataType)));

    HIP_CALL(hipMemcpy(discountCurveGpu, inArgsHost.discountCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(repoCurveGpu, inArgsHost.repoCurve, numBonds * sizeof(bondsYieldTermStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(currDateGpu, inArgsHost.currDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(maturityDateGpu, inArgsHost.maturityDate, numBonds * sizeof(bondsDateStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds * sizeof(dataType), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(bondGpu, inArgsHost.bond, numBonds * sizeof(bondStruct), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds * sizeof(dataType), hipMemcpyHostToDevice));

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

    hipLaunchKernelGGL((getBondsResultsGpu), dim3(grid), dim3(threads), 0, 0, inArgs, results, numBonds);
    HIP_CALL(hipMemcpy(resultsGpu.dirtyPrice, dirtyPriceGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsGpu.accruedAmountCurrDate, accruedAmountCurrDateGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsGpu.cleanPrice, cleanPriceGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(resultsGpu.bondForwardVal, bondForwardValGpu, numBonds * sizeof(dataType), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());

    for(int i = 0; i < numBonds; ++i)
    {
        ASSERT_NEAR(resultsCpu.dirtyPrice[i], resultsGpu.dirtyPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.accruedAmountCurrDate[i], resultsGpu.accruedAmountCurrDate[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.cleanPrice[i], resultsGpu.cleanPrice[i], 1e-4f) << "where index = " << i;
        ASSERT_NEAR(resultsCpu.bondForwardVal[i], resultsGpu.bondForwardVal[i], 1e-4f) << "where index = " << i;
    }

    HIP_CALL(hipFree(discountCurveGpu));
    HIP_CALL(hipFree(repoCurveGpu));
    HIP_CALL(hipFree(currDateGpu));
    HIP_CALL(hipFree(maturityDateGpu));
    HIP_CALL(hipFree(bondCleanPriceGpu));
    HIP_CALL(hipFree(bondGpu));
    HIP_CALL(hipFree(dummyStrikeGpu));
    HIP_CALL(hipFree(dirtyPriceGpu));
    HIP_CALL(hipFree(accruedAmountCurrDateGpu));
    HIP_CALL(hipFree(cleanPriceGpu));
    HIP_CALL(hipFree(bondForwardValGpu));

    delete [] resultsCpu.dirtyPrice;
    delete [] resultsCpu.accruedAmountCurrDate;
    delete [] resultsCpu.cleanPrice;
    delete [] resultsCpu.bondForwardVal;
    delete [] resultsGpu.dirtyPrice;
    delete [] resultsGpu.accruedAmountCurrDate;
    delete [] resultsGpu.cleanPrice;
    delete [] resultsGpu.bondForwardVal;

    delete [] inArgsHost.discountCurve;
    delete [] inArgsHost.repoCurve;
    delete [] inArgsHost.currDate;
    delete [] inArgsHost.maturityDate;
    delete [] inArgsHost.bondCleanPrice;
    delete [] inArgsHost.bond;
    delete [] inArgsHost.dummyStrike;
}
#endif
