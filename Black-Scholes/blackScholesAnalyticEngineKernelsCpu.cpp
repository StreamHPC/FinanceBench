//blackScholesAnalyticEngineKernelsCpu.cu
//Scott Grauer-Gray
//Kernels for running black scholes using the analytic engine

//declarations for the kernels
#include "blackScholesAnalyticEngineKernelsCpu.h"

//global function to retrieve the output value for an option
void getOutValOptionCpu(optionInputStruct * options,
                        float * outputVals,
                        int numVals)
{
    //check if within current options
    for(int numOption = 0; numOption < numVals; ++numOption)
    {
        optionInputStruct threadOption = options[numOption];

        payoffStruct currPayoff;
        currPayoff.type = threadOption.type;
        currPayoff.strike = threadOption.strike;

        yieldTermStruct qTS;
        qTS.timeYearFraction = threadOption.t;
        qTS.forward = threadOption.q;

        yieldTermStruct rTS;
        rTS.timeYearFraction = threadOption.t;
        rTS.forward = threadOption.r;

        blackVolStruct volTS;
        volTS.timeYearFraction = threadOption.t;
        volTS.volatility = threadOption.vol;

        blackScholesMertStruct stochProcess;
        stochProcess.x0 = threadOption.spot;
        stochProcess.dividendTS = qTS;
        stochProcess.riskFreeTS = rTS;
        stochProcess.blackVolTS = volTS;

        optionStruct currOption;
        currOption.payoff = currPayoff;
        currOption.yearFractionTime = threadOption.t;
        currOption.pricingEngine = stochProcess;

        float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
        float dividendDiscount = getDiscountOnDividendYield(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
        float riskFreeDiscount = getDiscountOnRiskFreeRate(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
        float spot = currOption.pricingEngine.x0;

        float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

        //declare the blackCalcStruct
        blackCalcStruct blackCalc;

        //initialize the calculator
        initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sqrt(variance), riskFreeDiscount);

        //retrieve the results values
        float resultVal = getResultVal(blackCalc);

        //write the resulting value to global memory
        outputVals[numOption] = resultVal;
    }
}

void getOutValOptionOpenMP(optionInputStruct * options,
                           float * outputVals,
                           int numVals)
{
    //check if within current options
    #pragma omp parallel for
    for(int numOption = 0; numOption < numVals; ++numOption)
    {
        optionInputStruct threadOption = options[numOption];

        payoffStruct currPayoff;
        currPayoff.type = threadOption.type;
        currPayoff.strike = threadOption.strike;

        yieldTermStruct qTS;
        qTS.timeYearFraction = threadOption.t;
        qTS.forward = threadOption.q;

        yieldTermStruct rTS;
        rTS.timeYearFraction = threadOption.t;
        rTS.forward = threadOption.r;

        blackVolStruct volTS;
        volTS.timeYearFraction = threadOption.t;
        volTS.volatility = threadOption.vol;

        blackScholesMertStruct stochProcess;
        stochProcess.x0 = threadOption.spot;
        stochProcess.dividendTS = qTS;
        stochProcess.riskFreeTS = rTS;
        stochProcess.blackVolTS = volTS;

        optionStruct currOption;
        currOption.payoff = currPayoff;
        currOption.yearFractionTime = threadOption.t;
        currOption.pricingEngine = stochProcess;

        float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
        float dividendDiscount = getDiscountOnDividendYield(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
        float riskFreeDiscount = getDiscountOnRiskFreeRate(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
        float spot = currOption.pricingEngine.x0;

        float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

        //declare the blackCalcStruct
        blackCalcStruct blackCalc;

        //initialize the calculator
        initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sqrt(variance), riskFreeDiscount);

        //retrieve the results values
        float resultVal = getResultVal(blackCalc);

        //write the resulting value to global memory
        outputVals[numOption] = resultVal;
    }
}
