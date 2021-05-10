import math
import numpy as np
import pandas as pd
import scipy.optimize as sop
from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib as mpl

i: float = 0
min_RMSE: float = 100
class NormalInverseGaussian:
    S0: float
    K: float
    T: float
    r: float
    dividendRate: float
    optionType: str
    delta: float
    alpha: float
    beta: float
    options: type


    def __init__(self, S0, K, T, r, sigma, dividendRate, optionType, optionData):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dividendRate = dividendRate
        self.optionType = optionType
        self.options = optionData

    def calibrate(self):
        global i, min_RMSE
        i = 0
        min_RMSE = 100
        p0 = sop.brute(self.NIG_error_function_FFT,
                       ((0.1, .6, .1),
                        (0, 10, 1),
                        (-10, 10, 1)), finish=None)

        opt = sop.fmin(self.NIG_error_function_FFT, p0,
                       maxiter=500, maxfun=750,
                       xtol=0.000001, ftol=0.000001)

        print(opt)
        self.delta = opt[0]
        self.alpha = opt[1]
        self.beta = opt[2]
        return opt

    def NIG_error_function_FFT(self, p0):
        ''' Error Function for parameter calibration in NIG Model via
        Carr-Madan (1999) FFT approach.

        Parameters
        ==========
        sigma: float
            volatility of the brownian motion
        nu: float
            variance rate of the gamma time change
        theta: float
            drift in brownian motion with drift (can be negative)


        Returns
        =======
        RMSE: float
            root mean squared error
        '''
        global i, min_RMSE
        delta, alpha, beta = p0

        if (abs(beta+1)**2 > (alpha)**2 or (abs(beta)**2 > alpha**2)):
            return 500.0
        se = []
        for row, option in self.options.iterrows():
            T = option["TTM"]
            model_value = self.NIG_value_call_FFT_Bear(self.S0, option['Strike Price'], T,
                                             self.r, delta, alpha, beta, self.dividendRate)
            se.append((model_value - option['Premium']) ** 2)
        RMSE = math.sqrt(sum(se) / len(se))
        min_RMSE = min(min_RMSE, RMSE)
        if i % 50 == 0:
            print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
        i += 1
        return RMSE

    def NIG_characteristic_function(self, u, x0, T, r, delta, alpha, beta, dividendRate):
        ''' Valuation of European call option in NIG model via
                Lewis (2001) Fourier-based approach: characteristic function.
                Parameter definitions see function NIG_value_call_FFT. '''
        try:
            omega = -1 * delta * (math.sqrt((alpha*alpha) - (beta*beta)) - math.sqrt((alpha*alpha) - ((beta+1)**2)))
        except (Exception):
            print('uh oh')

        insidePart = beta + (1j * u)
        secondSqrt = ((alpha**2) - (insidePart**2))**.5
        firstSqrt = ((alpha**2)-(beta**2))**.5
        sqrtProduct = (delta*T*(firstSqrt-secondSqrt))
        value = np.exp((1j*u*omega*T)+sqrtProduct)

        #value = np.exp((1j * u * omega * T) + delta*T((math.sqrt(alpha**2 - beta**2) - math.sqrt(alpha**2 - (beta+(1j*u))**2))))
        return value

    #
    # Valuation by FFT
    #
    def NIG_value_call_FFT(self, S0, K, T, r, delta, alpha, beta, dividend):
        ''' Valuation of European call option in NIG model via
        Carr-Madan (1999) Fourier-based approach.

        Parameters
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            time-to-maturity (for t=0)
        r: float
            constant risk-free short rate
        delta: float

        alpha: float

        beta: float

        dividend: float
            dividend rate

        Returns
        =======
        call_value: float
            European call option present value
        '''
        k = math.log(K / S0)
        x0 = math.log(S0 / S0)
        g = 3  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.) ** -1
        eta = 2 * math.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)
        # Modificatons to Ensure Integrability
        if S0 >= 0.95 * K:  # ITM case
            a = 1.5
            #a=alpha
            v = vo - (a + 1) * 1j
            mod_char_fun = math.exp(-(r - dividend) * T) * self.NIG_characteristic_function(v,x0, T, r, delta, alpha, beta, dividend) \
                           / (a ** 2 + a - vo ** 2 + 1j * (2 * a + 1) * vo)
        else:  # OTM case
            a = 1.1
            #a=alpha
            v = (vo - 1j * a) - 1j
            mod_char_fun_1 = math.exp(-(r - dividend) * T) * (1 / (1 + 1j * (vo - 1j * a))
                                                              - math.exp((r - dividend) * T) /
                                                              (1j * (vo - 1j * a))
                                                              - self.NIG_characteristic_function(v,x0, T, r, delta, alpha, beta, dividend) /
                                                              ((vo - 1j * a) ** 2 - 1j * (vo - 1j * a)))
            v = (vo + 1j * a) - 1j
            mod_char_fun_2 = math.exp(-(r - dividend) * T) * (1 / (1 + 1j * (vo + 1j * a))
                                                              - math.exp((r - dividend) * T) /
                                                              (1j * (vo + 1j * a))
                                                              - self.NIG_characteristic_function(v,x0, T, r, delta, alpha, beta, dividend) /
                                                              ((vo + 1j * a) ** 2 - 1j * (vo + 1j * a)))

        # Numerical FFT Routine
        delt = np.zeros(N, dtype=np.float)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3
        if S0 >= 0.95 * K:
            fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
            payoff = (fft(fft_func)).real
            call_value_m = np.exp(-a * k) / math.pi * payoff
        else:
            fft_func = (np.exp(1j * b * vo) *
                        (mod_char_fun_1 - mod_char_fun_2) *
                        0.5 * eta * SimpsonW)
            payoff = (fft(fft_func)).real
            call_value_m = payoff / (np.sinh(a * k) * math.pi)
        pos = int((k + b) / eps)
        call_value = call_value_m[pos]
        return call_value * S0

    def generate_plot(self, opt, options, singleCalibration:bool = False, isNDX:bool=False):
        #
        # Calculating Model Prices
        #
        delta, alpha, beta = opt
        options['NIG Model'] = 0.0
        for row, option in options.iterrows():
            T = (option['Expiration Date of the Option'] - option['The Date of this Price']).days / 365.
            options.loc[row, 'NIG Model'] = self.NIG_value_call_FFT_Bear(self.S0, option['Strike Price'],
                                                                       T, self.r, delta, alpha, beta, self.dividendRate)

        #
        # Plotting
        #
        mats = sorted(set(options['Expiration Date of the Option']))
        options = options.set_index('Strike Price')
        if (isNDX):
            val = "NDX"
        else:
            val = "SPX"
        for i, mat in enumerate(mats):
            options[options['Expiration Date of the Option'] == mat][['Premium', 'NIG Model']]. \
                plot(style=['b-', 'ro'], title='%s' % str(mat)[:10],
                     grid=True)
            plt.ylabel('option value')
            if (singleCalibration):
                plt.savefig('./NIG Plots/'+val+'_NIG_Single_Exp_Calibration.pdf')
            else:
                plt.savefig('./NIG Plots/'+val+'_NIG_calibration_3_%s.pdf' % i)

    def NIG_value_call_FFT_Bear(self, S0, K, T, r, delta, alpha, beta, dividend):
        ''' Valuation of European call option in NIG model via
        Carr-Madan (1999) Fourier-based approach.

        Parameters
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            time-to-maturity (for t=0)
        r: float
            constant risk-free short rate
        delta: float

        alpha: float

        beta: float

        dividend: float
            dividend rate

        Returns
        =======
        call_value: float
            European call option present value
        '''
        k = math.log(K / S0)
        x0 = math.log(S0 / S0)
        g = 3  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.) ** -1
        eta = 2 * math.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        a = 1.5
        #a=alpha
        v = vo - (a + 1) * 1j
        mod_char_fun = math.exp(-(r - dividend) * T) * self.NIG_characteristic_function(v,x0, T, r, delta, alpha, beta, dividend) \
                           / (a ** 2 + a - vo ** 2 + 1j * (2 * a + 1) * vo)

        # Numerical FFT Routine
        delt = np.zeros(N, dtype=np.float)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3

        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
        payoff = (fft(fft_func)).real
        call_value_m = np.exp(-a * k) / math.pi * payoff

        pos = int((k + b) / eps)
        call_value = call_value_m[pos]
        return call_value * S0