//repoKernels.cu
//Scott Grauer-Gray
//July 6, 2012
//Repo kernels on the

#include "repoKernelsCpu.h"

void getRepoResultsCpu(inArgsStruct inArgs,
                       resultsStruct results,
                       int totNumRuns)
{
    for(int repoNum = 0; repoNum < totNumRuns; ++repoNum)
    {
        int numLegs = getNumCashFlows(inArgs, repoNum);
        cashFlowsStruct cashFlows;
        cashFlows.legs = (couponStruct *)malloc(numLegs * sizeof(couponStruct));

        cashFlows.intRate.dayCounter = USE_EXACT_DAY;
        cashFlows.intRate.rate = inArgs.bond[repoNum].rate;
        cashFlows.intRate.freq = ANNUAL_FREQ;
        cashFlows.intRate.comp = SIMPLE_INTEREST;
        cashFlows.dayCounter = USE_EXACT_DAY;
        cashFlows.nominal = 100.0;

        //repoDateStruct currPaymentDate;
        repoDateStruct currStartDate = advanceDate(inArgs.bond[repoNum].maturityDate, (numLegs - 1) * -6);
        repoDateStruct currEndDate = advanceDate(currStartDate, 6);

        for(int cashFlowNum = 0; cashFlowNum < numLegs - 1; ++cashFlowNum)
        {
            cashFlows.legs[cashFlowNum].paymentDate = currEndDate;
            cashFlows.legs[cashFlowNum].accrualStartDate = currStartDate;
            cashFlows.legs[cashFlowNum].accrualEndDate = currEndDate;
            cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

            currStartDate = currEndDate;
            currEndDate = advanceDate(currEndDate, 6);
        }

        cashFlows.legs[numLegs - 1].paymentDate = inArgs.bond[repoNum].maturityDate;
        cashFlows.legs[numLegs - 1].accrualStartDate = inArgs.settlementDate[repoNum];
        cashFlows.legs[numLegs - 1].accrualEndDate = inArgs.settlementDate[repoNum];
        cashFlows.legs[numLegs - 1].amount = 100.0;

        results.bondForwardVal[repoNum] = getBondYield(
                                             inArgs.bondCleanPrice[repoNum],
                                             USE_EXACT_DAY,
                                             COMPOUNDED_INTEREST,
                                             2.0,
                                             inArgs.settlementDate[repoNum],
                                             ACCURACY,
                                             100,
                                             inArgs, repoNum, cashFlows, numLegs
                                          );
        inArgs.discountCurve[repoNum].forward = results.bondForwardVal[repoNum];
        results.dirtyPrice[repoNum] = getDirtyPrice(inArgs, repoNum, cashFlows, numLegs);
        results.accruedAmountSettlement[repoNum] = getAccruedAmount(inArgs, inArgs.settlementDate[repoNum], repoNum, cashFlows, numLegs);
        results.accruedAmountDeliveryDate[repoNum] = getAccruedAmount(inArgs, inArgs.deliveryDate[repoNum], repoNum, cashFlows, numLegs);
        results.cleanPrice[repoNum] = results.dirtyPrice[repoNum] - results.accruedAmountSettlement[repoNum];
        results.forwardSpotIncome[repoNum] = fixedRateBondForwardSpotIncome(inArgs, repoNum, cashFlows, numLegs);
        results.underlyingBondFwd[repoNum] = results.forwardSpotIncome[repoNum] / repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);
        dataType forwardVal = (results.dirtyPrice[repoNum] - results.forwardSpotIncome[repoNum]) / repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);

        results.repoNpv[repoNum] = (forwardVal - inArgs.dummyStrike[repoNum]) * repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);
        results.repoCleanForwardPrice[repoNum] = forwardVal - getAccruedAmount(inArgs, inArgs.repoDeliveryDate[repoNum], repoNum, cashFlows, numLegs);
        results.repoDirtyForwardPrice[repoNum] = forwardVal;
        results.repoImpliedYield[repoNum] = getImpliedYield(inArgs, inArgs.dummyStrike[repoNum], results.dirtyPrice[repoNum], results.forwardSpotIncome[repoNum], repoNum);
        results.marketRepoRate[repoNum] = getMarketRepoRate(
                                             inArgs.repoDeliveryDate[repoNum],
                                             SIMPLE_INTEREST,
                                             1.0,
                                             inArgs.settlementDate[repoNum],
                                             inArgs, repoNum
                                          );

        free(cashFlows.legs);
    }
}

void getRepoResultsOpenMP(inArgsStruct inArgs,
                          resultsStruct results,
                          int totNumRuns)
{
    #pragma omp parallel for
    for(int repoNum = 0; repoNum < totNumRuns; ++repoNum)
    {
        int numLegs = getNumCashFlows(inArgs, repoNum);
        cashFlowsStruct cashFlows;
        cashFlows.legs = (couponStruct *)malloc(numLegs * sizeof(couponStruct));

        cashFlows.intRate.dayCounter = USE_EXACT_DAY;
        cashFlows.intRate.rate = inArgs.bond[repoNum].rate;
        cashFlows.intRate.freq = ANNUAL_FREQ;
        cashFlows.intRate.comp = SIMPLE_INTEREST;
        cashFlows.dayCounter = USE_EXACT_DAY;
        cashFlows.nominal = 100.0;

        //repoDateStruct currPaymentDate;
        repoDateStruct currStartDate = advanceDate(inArgs.bond[repoNum].maturityDate, (numLegs - 1) * -6);
        repoDateStruct currEndDate = advanceDate(currStartDate, 6);

        for(int cashFlowNum = 0; cashFlowNum < numLegs - 1; ++cashFlowNum)
        {
            cashFlows.legs[cashFlowNum].paymentDate = currEndDate;
            cashFlows.legs[cashFlowNum].accrualStartDate = currStartDate;
            cashFlows.legs[cashFlowNum].accrualEndDate = currEndDate;
            cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

            currStartDate = currEndDate;
            currEndDate = advanceDate(currEndDate, 6);
        }

        cashFlows.legs[numLegs - 1].paymentDate = inArgs.bond[repoNum].maturityDate;
        cashFlows.legs[numLegs - 1].accrualStartDate = inArgs.settlementDate[repoNum];
        cashFlows.legs[numLegs - 1].accrualEndDate = inArgs.settlementDate[repoNum];
        cashFlows.legs[numLegs - 1].amount = 100.0;

        results.bondForwardVal[repoNum] = getBondYield(
                                             inArgs.bondCleanPrice[repoNum],
                                             USE_EXACT_DAY,
                                             COMPOUNDED_INTEREST,
                                             2.0,
                                             inArgs.settlementDate[repoNum],
                                             ACCURACY,
                                             100,
                                             inArgs, repoNum, cashFlows, numLegs
                                          );
        inArgs.discountCurve[repoNum].forward = results.bondForwardVal[repoNum];
        results.dirtyPrice[repoNum] = getDirtyPrice(inArgs, repoNum, cashFlows, numLegs);
        results.accruedAmountSettlement[repoNum] = getAccruedAmount(inArgs, inArgs.settlementDate[repoNum], repoNum, cashFlows, numLegs);
        results.accruedAmountDeliveryDate[repoNum] = getAccruedAmount(inArgs, inArgs.deliveryDate[repoNum], repoNum, cashFlows, numLegs);
        results.cleanPrice[repoNum] = results.dirtyPrice[repoNum] - results.accruedAmountSettlement[repoNum];
        results.forwardSpotIncome[repoNum] = fixedRateBondForwardSpotIncome(inArgs, repoNum, cashFlows, numLegs);
        results.underlyingBondFwd[repoNum] = results.forwardSpotIncome[repoNum] / repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);
        dataType forwardVal = (results.dirtyPrice[repoNum] - results.forwardSpotIncome[repoNum]) / repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);

        results.repoNpv[repoNum] = (forwardVal - inArgs.dummyStrike[repoNum]) * repoYieldTermStructureDiscount(inArgs.repoCurve[repoNum], inArgs.repoDeliveryDate[repoNum]);
        results.repoCleanForwardPrice[repoNum] = forwardVal - getAccruedAmount(inArgs, inArgs.repoDeliveryDate[repoNum], repoNum, cashFlows, numLegs);
        results.repoDirtyForwardPrice[repoNum] = forwardVal;
        results.repoImpliedYield[repoNum] = getImpliedYield(inArgs, inArgs.dummyStrike[repoNum], results.dirtyPrice[repoNum], results.forwardSpotIncome[repoNum], repoNum);
        results.marketRepoRate[repoNum] = getMarketRepoRate(
                                             inArgs.repoDeliveryDate[repoNum],
                                             SIMPLE_INTEREST,
                                             1.0,
                                             inArgs.settlementDate[repoNum],
                                             inArgs, repoNum
                                          );

        free(cashFlows.legs);
    }
}
