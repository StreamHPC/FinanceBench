void initOptions(optionInputStruct * values,
                 int numVals)
{
    for(int numOption = 0; numOption < numVals; ++numOption)
    {
        if((numOption % NUM_DIFF_SETTINGS) == 0)
        {
            optionInputStruct currVal = { CALL,  40.00,  42.00, 0.08, 0.04, 0.75, 0.35,  5.0975, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 1)
        {
            optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  0.0205, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 2)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 3)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  9.9413, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 4)
        {
            optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25,  0.3150, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 5)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 6)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25, 10.3556, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 7)
        {
            optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35,  0.9474, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 8)
        {
            optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 9)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35, 11.1381, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 10)
        {
            optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15,  0.8069, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 11)
        {
            optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 12)
        {
            optionInputStruct currVal =  { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15, 10.5769, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 13)
        {
            optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25,  2.7026, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 14)
        {
            optionInputStruct currVal =   { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 15)
        {
            optionInputStruct currVal =   { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25, 12.7857, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 16)
        {
            optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35,  4.9329, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 17)
        {
            optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 18)
        {
            optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35, 15.3086, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 19)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  9.9210, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 20)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 21)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  0.0408, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 22)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25, 10.2155, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 23)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 24)
        {
            optionInputStruct currVal =    { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25,  0.4551, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 25)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35, 10.8479, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 26)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 27)
        {
            optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35,  1.2376, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 28)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15, 10.3192, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 29)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 30)
        {
            optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15,  1.0646, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 31)
        {
            optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25, 12.2149, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 32)
        {
            optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 33)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25,  3.2734, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 34)
        {
            optionInputStruct currVal =   { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35, 14.4452, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 35)
        {
            optionInputStruct currVal =  { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
            values[numOption] = currVal;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 36)
        {
            optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35,  5.7963, 1.0e-4};
            values[numOption] = currVal;
        }
    }
}

void initOptions(char * type,
                 float * strike,
                 float * spot,
                 float * q,
                 float * r,
                 float * t,
                 float * vol,
                 float * value,
                 float * tol,
                 int numVals)
{
    for(int numOption = 0; numOption < numVals; ++numOption)
    {
        if((numOption % NUM_DIFF_SETTINGS) == 0)
        {
            type[numOption] = CALL;
            strike[numOption] = 40.00f;
            spot[numOption] = 42.00f;
            q[numOption] = 0.08f;
            r[numOption] = 0.04f;
            t[numOption] = 0.75f;
            vol[numOption] = 0.35f;
            value[numOption] = 5.0975f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 1)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 0.0205f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 2)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 1.8734f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 3)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 9.9413f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 4)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 0.3150f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 5)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 3.1217f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 6)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 10.3556f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 7)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 0.9474f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 8)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 4.3693f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 9)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 11.1381f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 10)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 0.8069f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 11)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 4.0232f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 12)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 10.5769f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 13)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 2.7026f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 14)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 6.6997f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 15)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 12.7857f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 16)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 4.9329f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 17)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 9.3679f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 18)
        {
            type[numOption] = CALL;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 15.3086f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 19)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 9.9210f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 20)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 1.8734f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 21)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.15f;
            value[numOption] = 0.0408f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 22)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 10.2155f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 23)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 3.1217f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 24)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.25f;
            value[numOption] = 0.4551f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 25)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 10.8479f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 26)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 4.3693f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 27)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.10f;
            vol[numOption] = 0.35f;
            value[numOption] = 1.2376f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 28)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 10.3192f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 29)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 4.0232f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 30)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.15f;
            value[numOption] = 1.0646f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 31)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 12.2149f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 32)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 6.6997f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 33)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.25f;
            value[numOption] = 3.2734f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 34)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 90.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 14.4452f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 35)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 100.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 9.3679f;
            tol[numOption] = 1.0e-4;
        }
        if((numOption % NUM_DIFF_SETTINGS) == 36)
        {
            type[numOption] = PUT;
            strike[numOption] = 100.00f;
            spot[numOption] = 110.00f;
            q[numOption] = 0.10f;
            r[numOption] = 0.10f;
            t[numOption] = 0.50f;
            vol[numOption] = 0.35f;
            value[numOption] = 5.7963f;
            tol[numOption] = 1.0e-4;
        }
    }
}
