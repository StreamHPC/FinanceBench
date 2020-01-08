//bondsKernels.cu
//Scott Grauer-Gray
//Bonds kernels to run on the

#include "bondsKernelsCpu.h"

void getBondsResultsCpu(inArgsStruct inArgs,
                        resultsStruct results,
                        int totNumRuns)
{
    for(int bondNum = 0; bondNum < totNumRuns; ++bondNum)
    {
        int numLegs  = getNumCashFlows(inArgs, bondNum);
        cashFlowsStruct cashFlows;
        cashFlows.legs = (couponStruct *)malloc(numLegs * sizeof(couponStruct));

        cashFlows.intRate.dayCounter = USE_EXACT_DAY;
        cashFlows.intRate.rate = inArgs.bond[bondNum].rate;
        cashFlows.intRate.freq = ANNUAL_FREQ;
        cashFlows.intRate.comp = SIMPLE_INTEREST;
        cashFlows.dayCounter = USE_EXACT_DAY;
        cashFlows.nominal = 100.0;

        //bondsDateStruct currPaymentDate;
        bondsDateStruct currStartDate = advanceDate(inArgs.bond[bondNum].maturityDate, (numLegs - 1) * -6);;
        bondsDateStruct currEndDate = advanceDate(currStartDate, 6);

        for(int cashFlowNum = 0; cashFlowNum < numLegs - 1; ++cashFlowNum)
        {
            cashFlows.legs[cashFlowNum].paymentDate = currEndDate;
            cashFlows.legs[cashFlowNum].accrualStartDate = currStartDate;
            cashFlows.legs[cashFlowNum].accrualEndDate = currEndDate;
            cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

            currStartDate = currEndDate;
            currEndDate = advanceDate(currEndDate, 6);
        }

        cashFlows.legs[numLegs - 1].paymentDate = inArgs.bond[bondNum].maturityDate;
        cashFlows.legs[numLegs - 1].accrualStartDate = inArgs.currDate[bondNum];
        cashFlows.legs[numLegs - 1].accrualEndDate = inArgs.currDate[bondNum];
        cashFlows.legs[numLegs - 1].amount = 100.0;

        results.bondForwardVal[bondNum] = getBondYield(
                                              inArgs.bondCleanPrice[bondNum],
                                              USE_EXACT_DAY, COMPOUNDED_INTEREST,
                                              2.0, inArgs.currDate[bondNum],
                                              ACCURACY, 100, inArgs, bondNum,
                                              cashFlows, numLegs
                                          );
        inArgs.discountCurve[bondNum].forward = results.bondForwardVal[bondNum];
        results.dirtyPrice[bondNum] = getDirtyPrice(inArgs, bondNum, cashFlows, numLegs);
        results.accruedAmountCurrDate[bondNum] = getAccruedAmount(inArgs, inArgs.currDate[bondNum], bondNum, cashFlows, numLegs);
        results.cleanPrice[bondNum] = results.dirtyPrice[bondNum] - results.accruedAmountCurrDate[bondNum];

        free(cashFlows.legs);
    }
}

void getBondsResultsOpenMP(inArgsStruct inArgs,
                           resultsStruct results,
                           int totNumRuns)
{
    #pragma omp parallel for
    for(int bondNum = 0; bondNum < totNumRuns; ++bondNum)
    {
        int numLegs  = getNumCashFlows(inArgs, bondNum);
        cashFlowsStruct cashFlows;
        cashFlows.legs = (couponStruct *)malloc(numLegs * sizeof(couponStruct));

        cashFlows.intRate.dayCounter = USE_EXACT_DAY;
        cashFlows.intRate.rate = inArgs.bond[bondNum].rate;
        cashFlows.intRate.freq = ANNUAL_FREQ;
        cashFlows.intRate.comp = SIMPLE_INTEREST;
        cashFlows.dayCounter = USE_EXACT_DAY;
        cashFlows.nominal = 100.0;

        //bondsDateStruct currPaymentDate;
        bondsDateStruct currStartDate = advanceDate(inArgs.bond[bondNum].maturityDate, (numLegs - 1) * -6);;
        bondsDateStruct currEndDate = advanceDate(currStartDate, 6);

        for(int cashFlowNum = 0; cashFlowNum < numLegs - 1; ++cashFlowNum)
        {
            cashFlows.legs[cashFlowNum].paymentDate = currEndDate;
            cashFlows.legs[cashFlowNum].accrualStartDate = currStartDate;
            cashFlows.legs[cashFlowNum].accrualEndDate = currEndDate;
            cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

            currStartDate = currEndDate;
            currEndDate = advanceDate(currEndDate, 6);
        }

        cashFlows.legs[numLegs - 1].paymentDate = inArgs.bond[bondNum].maturityDate;
        cashFlows.legs[numLegs - 1].accrualStartDate = inArgs.currDate[bondNum];
        cashFlows.legs[numLegs - 1].accrualEndDate = inArgs.currDate[bondNum];
        cashFlows.legs[numLegs - 1].amount = 100.0;

        results.bondForwardVal[bondNum] = getBondYield(
                                              inArgs.bondCleanPrice[bondNum],
                                              USE_EXACT_DAY, COMPOUNDED_INTEREST,
                                              2.0, inArgs.currDate[bondNum],
                                              ACCURACY, 100, inArgs, bondNum,
                                              cashFlows, numLegs
                                          );
        inArgs.discountCurve[bondNum].forward = results.bondForwardVal[bondNum];
        results.dirtyPrice[bondNum] = getDirtyPrice(inArgs, bondNum, cashFlows, numLegs);
        results.accruedAmountCurrDate[bondNum] = getAccruedAmount(inArgs, inArgs.currDate[bondNum], bondNum, cashFlows, numLegs);
        results.cleanPrice[bondNum] = results.dirtyPrice[bondNum] - results.accruedAmountCurrDate[bondNum];

        free(cashFlows.legs);
    }
}
