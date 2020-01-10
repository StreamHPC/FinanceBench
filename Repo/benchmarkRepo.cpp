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
#include "repoKernels.cuh"

#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

    // Warm-up
    /*for(size_t i = 0; i < warmup_size; i++)
    {
        getRepoResultsCpu(inArgsHost, resultsHost, numRepos);
    }*/

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getRepoResultsCpu(inArgsHost, resultsHost, numRepos);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountSettlement);
    free(resultsHost.accruedAmountDeliveryDate);
    free(resultsHost.cleanPrice);
    free(resultsHost.forwardSpotIncome);
    free(resultsHost.underlyingBondFwd);
    free(resultsHost.repoNpv);
    free(resultsHost.repoCleanForwardPrice);
    free(resultsHost.repoDirtyForwardPrice);
    free(resultsHost.repoImpliedYield);
    free(resultsHost.marketRepoRate);
    free(resultsHost.bondForwardVal);

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

void runBenchmarkOpenMP(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

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

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountSettlement);
    free(resultsHost.accruedAmountDeliveryDate);
    free(resultsHost.cleanPrice);
    free(resultsHost.forwardSpotIncome);
    free(resultsHost.underlyingBondFwd);
    free(resultsHost.repoNpv);
    free(resultsHost.repoCleanForwardPrice);
    free(resultsHost.repoDirtyForwardPrice);
    free(resultsHost.repoImpliedYield);
    free(resultsHost.marketRepoRate);
    free(resultsHost.bondForwardVal);

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

void runBenchmarkCudaV1(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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
    CUDA_CALL(cudaDeviceSynchronize());

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getRepoResultsGpu<<<gridDim, blockDim>>>(inArgsGpu, resultsGpu, numRepos);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        getRepoResultsGpu<<<gridDim, blockDim>>>(inArgsGpu, resultsGpu, numRepos);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
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

void runBenchmarkCudaV2(benchmark::State& state,
                        size_t size)
{
    int numRepos = size;

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

    resultsStruct resultsHost;
    resultsHost.dirtyPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountSettlement = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.accruedAmountDeliveryDate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.cleanPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.forwardSpotIncome = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.underlyingBondFwd = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoNpv = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoCleanForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoDirtyForwardPrice = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.repoImpliedYield = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.marketRepoRate = (dataType *)malloc(numRepos * sizeof(dataType));
    resultsHost.bondForwardVal = (dataType *)malloc(numRepos * sizeof(dataType));

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

    dim3 blockDim(256, 1);
    dim3 gridDim((size_t)ceil((dataType)numRepos / (dataType)blockDim.x), 1);

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        getRepoResultsGpu<<<gridDim, blockDim>>>(inArgsGpu, resultsGpu, numRepos);
        CUDA_CALL(cudaPeekAtLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CUDA_CALL(cudaMemcpy((inArgsGpu.discountCurve), inArgsHost.discountCurve, numRepos * sizeof(repoYieldTermStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.repoCurve), inArgsHost.repoCurve, numRepos * sizeof(repoYieldTermStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.settlementDate), inArgsHost.settlementDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.deliveryDate), inArgsHost.deliveryDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.maturityDate), inArgsHost.maturityDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.repoDeliveryDate), inArgsHost.repoDeliveryDate, numRepos * sizeof(repoDateStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.bondCleanPrice), inArgsHost.bondCleanPrice, numRepos * sizeof(dataType), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.bond), inArgsHost.bond, numRepos * sizeof(bondStruct), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy((inArgsGpu.dummyStrike), inArgsHost.dummyStrike, numRepos * sizeof(dataType), cudaMemcpyHostToDevice));

        getRepoResultsGpu<<<gridDim, blockDim>>>(inArgsGpu, resultsGpu, numRepos);
        CUDA_CALL(cudaPeekAtLastError());

        CUDA_CALL(cudaMemcpy(resultsHost.dirtyPrice, (resultsGpu.dirtyPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.accruedAmountSettlement, (resultsGpu.accruedAmountSettlement), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.accruedAmountDeliveryDate, (resultsGpu.accruedAmountDeliveryDate), numRepos * sizeof(dataType),cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.cleanPrice, (resultsGpu.cleanPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.forwardSpotIncome, (resultsGpu.forwardSpotIncome), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.underlyingBondFwd, (resultsGpu.underlyingBondFwd), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.repoNpv, (resultsGpu.repoNpv), numRepos *sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.repoCleanForwardPrice, (resultsGpu.repoCleanForwardPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.repoDirtyForwardPrice, (resultsGpu.repoDirtyForwardPrice), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.repoImpliedYield, (resultsGpu.repoImpliedYield), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.marketRepoRate, (resultsGpu.marketRepoRate), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(resultsHost.bondForwardVal, (resultsGpu.bondForwardVal), numRepos * sizeof(dataType), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
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

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountSettlement);
    free(resultsHost.accruedAmountDeliveryDate);
    free(resultsHost.cleanPrice);
    free(resultsHost.forwardSpotIncome);
    free(resultsHost.underlyingBondFwd);
    free(resultsHost.repoNpv);
    free(resultsHost.repoCleanForwardPrice);
    free(resultsHost.repoDirtyForwardPrice);
    free(resultsHost.repoImpliedYield);
    free(resultsHost.marketRepoRate);
    free(resultsHost.bondForwardVal);

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

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", 10, "number of iterations");
    parser.set_optional<int>("seed", "seed", 123, "seed for RNG");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    const int seed = parser.get<int>("seed");
    srand(seed);

    int runtime_version;
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            ("repo (CPU)"),
            [=](benchmark::State& state) { runBenchmarkCpu(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("repo (OpenMP)"),
            [=](benchmark::State& state) { runBenchmarkOpenMP(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("repoCuda (Compute Only)"),
            [=](benchmark::State& state) { runBenchmarkCudaV1(state, size); }
        ),
        benchmark::RegisterBenchmark(
            ("repoCuda (+ Transfers)"),
            [=](benchmark::State& state) { runBenchmarkCudaV2(state, size); }
        ),
    };

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
