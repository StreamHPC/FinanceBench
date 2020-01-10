//blackScholesAnalyticEngineKernels.cu
//Scott Grauer-Gray
//Kernels for running black scholes using the analytic engine

//declarations for the kernels
#include "blackScholesAnalyticEngineKernels.cuh"

//needed for the constants in the error function
//#include "errorFunctConsts.h"

//device kernel to run the operator function in cumulative normal distribution
HOST_DEVICE float cumNormDistOpV2(normalDistStruct normDist,
                                  float z)
{
    z = (z - normDist.average) / normDist.sigma;
    //float result = 0.5 * (1.0 + errorFunct(normDist, z * M_SQRT_2));
    float result = 0.5f * (1.0f + erf(z * M_SQRT_2));
    return result;
}

//device function to initialize the cumulative normal distribution structure
HOST_DEVICE void initCumNormDistV2(normalDistStruct& currCumNormDist)
{
    currCumNormDist.average = 0.0f;
    currCumNormDist.sigma = 1.0f;
    currCumNormDist.normalizationFactor = M_SQRT_2 * M_1_SQRTPI / currCumNormDist.sigma;
    currCumNormDist.derNormalizationFactor = currCumNormDist.sigma * currCumNormDist.sigma;
    currCumNormDist.denominator = 2.0f * currCumNormDist.derNormalizationFactor;
}

//device function to initialize variable in the black calculator
HOST_DEVICE void initBlackCalcVarsV2(blackCalcStruct& blackCalculator,
                                     payoffStruct payoff)
{
    blackCalculator.d1 = log(blackCalculator.forward / blackCalculator.strike) /
                         blackCalculator.stdDev + 0.5f * blackCalculator.stdDev;
    blackCalculator.d2 = blackCalculator.d1 - blackCalculator.stdDev;

    //initialize the cumulative normal distribution structure
    normalDistStruct currCumNormDist;
    initCumNormDistV2(currCumNormDist);

    blackCalculator.cum_d1 = cumNormDistOpV2(currCumNormDist, blackCalculator.d1);
    blackCalculator.cum_d2 = cumNormDistOpV2(currCumNormDist, blackCalculator.d2);
    blackCalculator.n_d1 = cumNormDistDeriv(currCumNormDist, blackCalculator.d1);
    blackCalculator.n_d2 = cumNormDistDeriv(currCumNormDist, blackCalculator.d2);

    blackCalculator.x = payoff.strike;
    blackCalculator.DxDstrike = 1.0f;

    // the following one will probably disappear as soon as
    // super-share will be properly handled
    blackCalculator.DxDs = 0.0f;

    // this part is always executed.
    // in case of plain-vanilla payoffs, it is also the only part
    // which is executed.
    switch(payoff.type)
    {
        case CALL:
            blackCalculator.alpha     =  blackCalculator.cum_d1;//  N(d1)
            blackCalculator.DalphaDd1 =  blackCalculator.n_d1;//  n(d1)
            blackCalculator.beta      = -1.0f * blackCalculator.cum_d2;// -N(d2)
            blackCalculator.DbetaDd2  = -1.0f * blackCalculator.n_d2;// -n(d2)
        break;
        case PUT:
            blackCalculator.alpha     = -1.0f + blackCalculator.cum_d1;// -N(-d1)
            blackCalculator.DalphaDd1 =  blackCalculator.n_d1;//  n( d1)
            blackCalculator.beta      =  1.0f - blackCalculator.cum_d2;//  N(-d2)
            blackCalculator.DbetaDd2  = -1.0f * blackCalculator.n_d2;// -n( d2)
        break;
    }
}

//device function to initialize the black calculator
HOST_DEVICE void initBlackCalculatorV2(blackCalcStruct& blackCalc,
                                       payoffStruct payoff,
                                       float forwardPrice,
                                       float stdDev,
                                       float riskFreeDiscount)
{
    blackCalc.strike = payoff.strike;
    blackCalc.forward = forwardPrice;
    blackCalc.stdDev = stdDev;
    blackCalc.discount = riskFreeDiscount;
    blackCalc.variance = stdDev * stdDev;

    initBlackCalcVarsV2(blackCalc, payoff);
}

//global function to retrieve the output value for an option
__global__ void getOutValOption(optionInputStruct * options,
                                float * outputVals,
                                int numVals)
{
	int optionNum = blockIdx.x * blockDim.x + threadIdx.x;

	//check if within current options
	if(optionNum < numVals)
	{
		optionInputStruct threadOption = options[optionNum];

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
		outputVals[optionNum] = resultVal;
	}
}

//global function to retrieve the output value for an option
__global__ void getOutValOptionOpt(optionInputStruct * options,
                                   float * outputVals,
                                   int numVals)
{
	int optionNum = blockIdx.x * blockDim.x + threadIdx.x;

	//check if within current options
	if(optionNum < numVals)
	{
		optionInputStruct threadOption = options[optionNum];

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
		initBlackCalculatorV2(blackCalc, currOption.payoff, forwardPrice, sqrt(variance), riskFreeDiscount);

		//retrieve the results values
		float resultVal = getResultVal(blackCalc);

		//write the resulting value to global memory
		outputVals[optionNum] = resultVal;
	}
}
