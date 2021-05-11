import math
import numpy as np
import pandas as pd
import scipy.optimize as sop
from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib as mpl

i: float = 0
min_RMSE: float = 100
class VarianceGamma:
    S0: float
    K: float
    T: float
    r: float
    dividendRate: float
    optionType: str
    sigma: float
    nu: float
    theta: float
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
        p0 = sop.brute(self.VG_error_function_FFT,
                       ((0.075, 0.17, 0.05),
                        (0.05, 0.5, 0.05),
                        (-0.8, 0.3, 0.05)), finish=None)

        opt = sop.fmin(self.VG_error_function_FFT, p0,
                       maxiter=500, maxfun=750,
                       xtol=0.000001, ftol=0.000001)

        print(opt)
        self.sigma = opt[0]
        self.nu = opt[1]
        self.theta = opt[2]
        return opt

    def VG_error_function_FFT(self, p0):
        ''' Error Function for parameter calibration in VG Model via
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
        sigma, nu, theta = p0
        #Theta can be < 0 (Pg 9 of VG paper)
        if sigma < 0.0 or nu < 0.0:
            return 500.0
        se = []
        for row, option in self.options.iterrows():
            T = option["TTM"]
            model_value = self.VG_value_call_FFT_Bear(self.S0, option['Strike Price'], T,
                                             self.r, sigma, nu, theta, self.dividendRate)
            se.append((model_value - option['Premium']) ** 2)
        RMSE = math.sqrt(sum(se) / len(se))
        min_RMSE = min(min_RMSE, RMSE)
        if i % 50 == 0:
            print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
        i += 1
        return RMSE

    def VG_characteristic_function(self, u, x0, T, r, sigma, nu, theta, dividendRate):
        ''' Valuation of European call option in VG model via
                Lewis (2001) Fourier-based approach: characteristic function.
                Parameter definitions see function VG_value_call_FFT. '''
        omega = (1/nu) * np.log(1 - (nu*theta) - (.5 * (sigma * sigma) * nu))
        value = np.exp(1j * u * omega * T) * pow((1-(1j*u*nu*theta) + (0.5*(sigma*sigma)*nu*(u*u))), -T/nu)
        return value

    #
    # Valuation by FFT
    #
    def VG_value_call_FFT(self, S0, K, T, r, sigma, nu, theta, dividend):
        ''' Valuation of European call option in VG model via
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
        sigma: float

        nu: float

        theta: float

        dividend: float
            dividend rate

        Returns
        =======
        call_value: float
            European call option present value
        '''
        k = math.log(K / S0)
        x0 = math.log(S0 / S0)
        g = 2  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.) ** -1
        eta = 2 * math.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)
        # Modificatons to Ensure Integrability
        if S0 >= 0.95 * K:  # ITM case
            alpha = 1.5
            v = vo - (alpha + 1) * 1j
            mod_char_fun = math.exp(-(r - dividend) * T) * self.VG_characteristic_function(v,x0, T, r, sigma, nu, theta, dividend) \
                           / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
        else:  # OTM case
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            mod_char_fun_1 = math.exp(-(r - dividend) * T) * (1 / (1 + 1j * (vo - 1j * alpha))
                                                              - math.exp((r - dividend) * T) /
                                                              (1j * (vo - 1j * alpha))
                                                              - self.VG_characteristic_function(v,x0, T, r, sigma, nu, theta, dividend) /
                                                              ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
            v = (vo + 1j * alpha) - 1j
            mod_char_fun_2 = math.exp(-(r - dividend) * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                                              - math.exp((r - dividend) * T) /
                                                              (1j * (vo + 1j * alpha))
                                                              - self.VG_characteristic_function(v,x0, T, r, sigma, nu, theta, dividend) /
                                                              ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))

        # Numerical FFT Routine
        delt = np.zeros(N, dtype=np.float)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3
        if S0 >= 0.95 * K:
            fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
            payoff = (fft(fft_func)).real
            call_value_m = np.exp(-alpha * k) / math.pi * payoff
        else:
            fft_func = (np.exp(1j * b * vo) *
                        (mod_char_fun_1 - mod_char_fun_2) *
                        0.5 * eta * SimpsonW)
            payoff = (fft(fft_func)).real
            call_value_m = payoff / (np.sinh(alpha * k) * math.pi)
        pos = int((k + b) / eps)
        call_value = call_value_m[pos]
        return call_value * S0

    def generate_plot(self, opt, options, singleCalibration:bool = False, isNDX:bool = False):
        #
        # Calculating Model Prices
        #
        sigma, nu, theta = opt
        options['VG Model'] = 0.0
        for row, option in options.iterrows():
            T = (option['Expiration Date of the Option'] - option['The Date of this Price']).days / 365.
            options.loc[row, 'VG Model'] = self.VG_value_call_FFT_Bear(self.S0, option['Strike Price'],
                                                                       T, self.r, sigma, nu, theta, self.dividendRate)

        #
        # Plotting
        #
        options['Residual Diff'] = 0.0
        options['Residual Diff'] = abs(options['Premium'] - options['VG Model'])
        print('\nAvg $ Diff: '+str(np.average(options['Residual Diff']))+'\n')
        print('Total $ Diff: '+str(np.sum(options['Residual Diff']))+'\n')
        print('Median $ Diff: '+str(np.median(options['Residual Diff']))+'\n')
        mats = sorted(set(options['Expiration Date of the Option']))
        options = options.set_index('Strike Price')
        if (isNDX):
            val = "NDX"
        else:
            val = "SPX"
        for i, mat in enumerate(mats):
            options[options['Expiration Date of the Option'] == mat][['Premium', 'VG Model']]. \
                plot(style=['b-', 'ro'], title='%s' % str(mat)[:10],
                     grid=True)
            plt.ylabel('option value')
            if (singleCalibration):
                plt.savefig('./VG Plots/'+val+'_VG_Single_Exp_Calibration.pdf')
            else:
                plt.savefig('./VG Plots/'+val+'_VG_calibration_3_%s.pdf' % i)

#
    # Valuation by FFT
    #
    def VG_value_call_FFT_Bear(self, S0, K, T, r, sigma, nu, theta, dividend):
        ''' Valuation of European call option in VG model via
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
        sigma: float

        nu: float

        theta: float

        dividend: float
            dividend rate

        Returns
        =======
        call_value: float
            European call option present value
        '''
        k = math.log(K / S0)
        x0 = math.log(S0 / S0)
        g = 2  # factor to increase accuracy
        N = g * 4096
        eps = (g * 150.) ** -1
        eta = 2 * math.pi / (N * eps)
        b = 0.5 * N * eps - k
        u = np.arange(1, N + 1, 1)
        vo = eta * (u - 1)

        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        mod_char_fun = math.exp(-(r - dividend) * T) * self.VG_characteristic_function(v,x0, T, r, sigma, nu, theta, dividend) \
                           / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)

        # Numerical FFT Routine
        delt = np.zeros(N, dtype=np.float)
        delt[0] = 1
        j = np.arange(1, N + 1, 1)
        SimpsonW = (3 + (-1) ** j - delt) / 3

        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
        payoff = (fft(fft_func)).real
        call_value_m = np.exp(-alpha * k) / math.pi * payoff

        pos = int((k + b) / eps)
        call_value = call_value_m[pos]
        return call_value * S0