#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <algorithm>

// Google Benchmark
#include "benchmark/benchmark.h"
#include "cmdparser.hpp"

#include "repoKernelsCpu.h"
#ifdef BUILD_HIP
#include "repoKernelsGpu.h"
#include <hip/hip_runtime.h>

#define HIP_CALL(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#endif

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024;
#endif

const unsigned int warmup_size = 5;

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

void runBenchmarkCpu(benchmark::State& state,
                     size_t size)
{
    int numRepos = size;

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = new dataType[numRepos];
    resultsHost.accruedAmountSettlement = new dataType[numRepos];
    resultsHost.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsHost.cleanPrice = new dataType[numRepos];
    resultsHost.forwardSpotIncome = new dataType[numRepos];
    resultsHost.underlyingBondFwd = new dataType[numRepos];
    resultsHost.repoNpv = new dataType[numRepos];
    resultsHost.repoCleanForwardPrice = new dataType[numRepos];
    resultsHost.repoDirtyForwardPrice = new dataType[numRepos];
    resultsHost.repoImpliedYield = new dataType[numRepos];
    resultsHost.marketRepoRate = new dataType[numRepos];
    resultsHost.bondForwardVal = new dataType[numRepos];

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getRepoResultsCpu(inArgsHost, resultsHost, numRepos);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getRepoResultsCpu(inArgsHost, resultsHost, numRepos);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    delete [] resultsHost.dirtyPrice;
    delete [] resultsHost.accruedAmountSettlement;
    delete [] resultsHost.accruedAmountDeliveryDate;
    delete [] resultsHost.cleanPrice;
    delete [] resultsHost.forwardSpotIncome;
    delete [] resultsHost.underlyingBondFwd;
    delete [] resultsHost.repoNpv;
    delete [] resultsHost.repoCleanForwardPrice;
    delete [] resultsHost.repoDirtyForwardPrice;
    delete [] resultsHost.repoImpliedYield;
    delete [] resultsHost.marketRepoRate;
    delete [] resultsHost.bondForwardVal;

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

void runBenchmarkOpenMP(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = new dataType[numRepos];
    resultsHost.accruedAmountSettlement = new dataType[numRepos];
    resultsHost.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsHost.cleanPrice = new dataType[numRepos];
    resultsHost.forwardSpotIncome = new dataType[numRepos];
    resultsHost.underlyingBondFwd = new dataType[numRepos];
    resultsHost.repoNpv = new dataType[numRepos];
    resultsHost.repoCleanForwardPrice = new dataType[numRepos];
    resultsHost.repoDirtyForwardPrice = new dataType[numRepos];
    resultsHost.repoImpliedYield = new dataType[numRepos];
    resultsHost.marketRepoRate = new dataType[numRepos];
    resultsHost.bondForwardVal = new dataType[numRepos];

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getRepoResultsOpenMP(inArgsHost, resultsHost, numRepos);
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getRepoResultsOpenMP(inArgsHost, resultsHost, numRepos);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    delete [] resultsHost.dirtyPrice;
    delete [] resultsHost.accruedAmountSettlement;
    delete [] resultsHost.accruedAmountDeliveryDate;
    delete [] resultsHost.cleanPrice;
    delete [] resultsHost.forwardSpotIncome;
    delete [] resultsHost.underlyingBondFwd;
    delete [] resultsHost.repoNpv;
    delete [] resultsHost.repoCleanForwardPrice;
    delete [] resultsHost.repoDirtyForwardPrice;
    delete [] resultsHost.repoImpliedYield;
    delete [] resultsHost.marketRepoRate;
    delete [] resultsHost.bondForwardVal;

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
void runBenchmarkHipV1(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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

    inArgsStruct inArgsGpu;
    resultsStruct resultsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

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

    HIP_CALL(hipMemcpyAsync((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipMemcpyAsync((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));
    HIP_CALL(hipStreamSynchronize(stream));

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        hipLaunchKernelGGL((getRepoResultsGpu), dim3(gridDim), dim3(blockDim), 0, stream, inArgsGpu, resultsGpu, numRepos);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL((getRepoResultsGpu), dim3(gridDim), dim3(blockDim), 0, stream, inArgsGpu, resultsGpu, numRepos);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
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
    HIP_CALL(hipStreamDestroy(stream));

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

void runBenchmarkHipV2(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = new dataType[numRepos];
    resultsHost.accruedAmountSettlement = new dataType[numRepos];
    resultsHost.accruedAmountDeliveryDate = new dataType[numRepos];
    resultsHost.cleanPrice = new dataType[numRepos];
    resultsHost.forwardSpotIncome = new dataType[numRepos];
    resultsHost.underlyingBondFwd = new dataType[numRepos];
    resultsHost.repoNpv = new dataType[numRepos];
    resultsHost.repoCleanForwardPrice = new dataType[numRepos];
    resultsHost.repoDirtyForwardPrice = new dataType[numRepos];
    resultsHost.repoImpliedYield = new dataType[numRepos];
    resultsHost.marketRepoRate = new dataType[numRepos];
    resultsHost.bondForwardVal = new dataType[numRepos];

    inArgsStruct inArgsGpu;
    resultsStruct resultsGpu;

    hipStream_t stream;
    HIP_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

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

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CALL(hipMemcpyAsync((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL((getRepoResultsGpu), dim3(gridDim), dim3(blockDim), 0, stream, inArgsGpu, resultsGpu, numRepos);
        HIP_CALL(hipPeekAtLastError());
        HIP_CALL(hipStreamSynchronize(stream));
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_CALL(hipMemcpyAsync((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), hipMemcpyHostToDevice, stream));
        HIP_CALL(hipMemcpyAsync((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), hipMemcpyHostToDevice, stream));

        hipLaunchKernelGGL((getRepoResultsGpu), dim3(gridDim), dim3(blockDim), 0, stream, inArgsGpu, resultsGpu, numRepos);
        HIP_CALL(hipPeekAtLastError());

        HIP_CALL(hipMemcpyAsync(resultsHost.dirtyPrice, (resultsGpu.dirtyPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.accruedAmountSettlement, (resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.accruedAmountDeliveryDate, (resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType),hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.cleanPrice, (resultsGpu.cleanPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.forwardSpotIncome, (resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.underlyingBondFwd, (resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.repoNpv, (resultsGpu.repoNpv), numRepos *sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.repoCleanForwardPrice, (resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.repoDirtyForwardPrice, (resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.repoImpliedYield, (resultsGpu.repoImpliedYield), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.marketRepoRate, (resultsGpu.marketRepoRate), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
		HIP_CALL(hipMemcpyAsync(resultsHost.bondForwardVal, (resultsGpu.bondForwardVal), numRepos * sizeof(dataType), hipMemcpyDeviceToHost, stream));
        HIP_CALL(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
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
    HIP_CALL(hipStreamDestroy(stream));

    delete [] resultsHost.dirtyPrice;
    delete [] resultsHost.accruedAmountSettlement;
    delete [] resultsHost.accruedAmountDeliveryDate;
    delete [] resultsHost.cleanPrice;
    delete [] resultsHost.forwardSpotIncome;
    delete [] resultsHost.underlyingBondFwd;
    delete [] resultsHost.repoNpv;
    delete [] resultsHost.repoCleanForwardPrice;
    delete [] resultsHost.repoDirtyForwardPrice;
    delete [] resultsHost.repoImpliedYield;
    delete [] resultsHost.marketRepoRate;
    delete [] resultsHost.bondForwardVal;

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

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.set_optional<int>("seed", "seed", 123, "seed for RNG");
    parser.set_optional<int>("device_id", "device_id", 0, "ID of GPU to run");
    parser.set_optional<bool>("run_cpu", "run_cpu", false, "Run single-threaded CPU version (slow)");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    const int seed = parser.get<int>("seed");
    const int device_id = parser.get<int>("device_id");
    const bool run_cpu = parser.get<bool>("run_cpu");
    srand(seed);

    #ifdef BUILD_HIP
    //int device_id;
    //HIP_CALL(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device_id));
    HIP_CALL(hipSetDevice(device_id));

    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;
    #endif

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("repo (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        #ifdef BUILD_HIP
        benchmark::RegisterBenchmark(
            ("repoHip (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkHipV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("repoHip (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkHipV2(state, size); }
        ),
        #endif
    };

    if(run_cpu)
    {
        benchmarks.insert(
            benchmarks.begin(),
            benchmark::RegisterBenchmark(
                ("repo (CPU)"),
                [=](benchmark::State& state) { runBenchmarkCpu(state, size); }
            )
        );
    }

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
