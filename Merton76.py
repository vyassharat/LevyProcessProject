import math
import numpy as np
import pandas as pd
import scipy.optimize as sop
from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib as mpl

i: float = 0
min_RMSE: float = 100
class Merton76:
    S0: float
    K: float
    T: float
    r: float
    dividendRate: float
    optionType: str
    sigma: float
    lamb: float
    mu: float
    delta: float
    options: type;


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
        p0 = sop.brute(self.M76_error_function_FFT,
                       ((0.075, 0.201, 0.025),
                        (0.10, 0.401, 0.1), (-0.5, 0.01, 0.1),
                        (0.10, 0.301, 0.1)), finish=None)

        # p0 = [0.15, 0.2, -0.3, 0.2]
        opt = sop.fmin(self.M76_error_function_FFT, p0,
                       maxiter=500, maxfun=750,
                       xtol=0.000001, ftol=0.000001)

        print(opt)
        return opt

    def M76_error_function_FFT(self, p0):
        ''' Error Function for parameter calibration in M76 Model via
        Carr-Madan (1999) FFT approach.

        Parameters
        ==========
        sigma: float
            volatility factor in diffusion term
        lamb: float
            jump intensity
        mu: float
            expected jump size
        delta: float
            standard deviation of jump

        Returns
        =======
        RMSE: float
            root mean squared error
        '''
        global i, min_RMSE
        sigma, lamb, mu, delta = p0
        if sigma < 0.0 or delta < 0.0 or lamb < 0.0:
            return 500.0
        se = []
        for row, option in self.options.iterrows():
            #T = (option['Maturity'] - option['Date']).days / 365.
            #TODO: Should self.S0 and self.r be the best way here?
            T = option["TTM"]
            model_value = self.M76_value_call_FFT(self.S0, option['Strike'], T,
                                             self.r, sigma, lamb, mu, delta)
            se.append((model_value - option['Call']) ** 2)
        RMSE = math.sqrt(sum(se) / len(se))
        min_RMSE = min(min_RMSE, RMSE)
        if i % 50 == 0:
            print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
        i += 1
        return RMSE

    def M76_characteristic_function(self, u, x0, T, r, sigma, lamb, mu, delta):
        ''' Valuation of European call option in M76 model via
        Lewis (2001) Fourier-based approach: characteristic function.

        Parameter definitions see function M76_value_call_INT. '''
        omega = x0 / T + r - 0.5 * sigma ** 2 \
                - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
        value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
                        lamb * (np.exp(1j * u * mu -
                                       u ** 2 * delta ** 2 * 0.5) - 1)) * T)
        return value

    #
    # Valuation by FFT
    #

    def M76_value_call_FFT(self, S0, K, T, r, sigma, lamb, mu, delta):
        ''' Valuation of European call option in M76 model via
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
            volatility factor in diffusion term
        lamb: float
            jump intensity
        mu: float
            expected jump size
        delta: float
            standard deviation of jump

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
            mod_char_fun = math.exp(-r * T) * self.M76_characteristic_function(
                v, x0, T, r, sigma, lamb, mu, delta) \
                           / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
        else:  # OTM case
            alpha = 1.1
            v = (vo - 1j * alpha) - 1j
            mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))
                                                 - math.exp(r * T) /
                                                 (1j * (vo - 1j * alpha))
                                                 - self.M76_characteristic_function(
                        v, x0, T, r, sigma, lamb, mu, delta) /
                                                 ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
            v = (vo + 1j * alpha) - 1j
            mod_char_fun_2 = math.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                                 - math.exp(r * T) /
                                                 (1j * (vo + 1j * alpha))
                                                 - self.M76_characteristic_function(
                        v, x0, T, r, sigma, lamb, mu, delta) /
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
