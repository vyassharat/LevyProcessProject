"""
Created on Mon Feb 15 09:15:50 2021

@author: svyas7
#Loosely followed from Derivatives Analytics With Python implementation
To incorporate the dividend yield, I loosely followed the formulas outlined below
https://www.macroption.com/black-scholes-formula/

I verified my answers by checking against https://goodcalculators.com/black-scholes-calculator/
where I was close to the greeks outlined at the bottom of the page.
"""

import math
from scipy.integrate import quad
from scipy.optimize import fsolve


class EuropeanOption:
    def __init__(self, S0, K, t, M, r, sigma, d, CP, C=0):
        self.S0 = S0
        self.K = K
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma
        self.d = d
        self.CP = CP
        self.C = C

    ''' PDF of standard normal random variable '''

    def dN(self, x):
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    ''' CDF of standard normal random variable'''

    def N(self, d):
        return quad(lambda x: self.dN(x), -20, d, limit=50)[0]

    '''BSM d1 function'''

    def d1f(self, St, K, t, T, r, sigma, d):
        d1 = (math.log(St / K) + (r - d + 0.5 * sigma ** 2)
              * (T - t)) / (sigma * math.sqrt(T - t))
        return d1;

    def BSM_call_value(self, St, K, t, T, r, sigma, d):
        '''
        Parameters
        ==========
        St: float
            stock/index level at time t
        K: float
            strike price
        t: float
            valuation date
        T: float
            date of maturity
        r: float
            constant, risk-free IR
        sigma: float
            implied vol
        '''

        d1 = self.d1f(St, K, t, T, r, sigma, d)
        d2 = d1 - sigma * math.sqrt(T - t)
        call_value = (St * math.exp(-d*(T - t)) * self.N(d1)) - math.exp(-r * (T - t)) * K * self.N(d2)
        return call_value

    def BSM_put_value(self, St, K, t, T, r, sigma, d):
        ''' Calculates Black-Scholes-Merton European put option value.
        Parameters
        ==========
        St : float
            stock/index level at time t
        K : float
            strike price
        t : float
            valuation date
        T : float
            date of maturity/time-to-maturity if t = 0; T > t
        r : float
            constant, risk-less short rate
        sigma : float
            volatility
        Returns
        =======
        put : float
            European put present value at t
        '''
        #put = (self.BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T - t)) * K)
        d1 = self.d1f(St, K, t, T, r, sigma, d)
        d2 = d1 - sigma * math.sqrt(T - t)
        put_value = math.exp(-r * (T - t)) * K * self.N(-1*d2) - (St * math.exp(-d*(T - t)) * self.N(-1*d1))
        return put_value

    def value(self):
        if(self.CP == 'Call'):
            return self.BSM_call_value(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
        else:
            return self.BSM_put_value(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)

    def imp_vol(self, sigma_est=0.2):
        def difference(sigma):
            self.sigma = sigma
            return self.value()-self.C

        iv = fsolve(difference, sigma_est)[0]
        return iv

    def delta(self):
        if(self.CP == 'Call'):
            d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
            delta = math.exp(-self.d* (self.M - self.t))*self.N(d1)
        else:
            d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
            delta = -math.exp(-self.d* (self.M - self.t)) * self.N(-d1)
        return delta

    def gamma(self):
        d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
        gamma = (math.exp(-self.d* (self.M - self.t)) * self.dN(d1)) / (self.S0 * self.sigma * math.sqrt(self.M-self.t))
        return gamma

    def vega(self):
        d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
        vega = math.exp(-self.d* (self.M - self.t)) * self.S0 * self.dN(d1) * math.sqrt(self.M - self.t)
        return vega

    #For this I tried to follow what was outlined in
    #http://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf
    def theta(self):
        d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
        d2 = d1 - self.sigma * math.sqrt(self.M - self.t)
        if(self.CP == 'Call'):
            theta=((-math.exp(self.d*(self.M-self.t))*self.S0 * self.dN(d1)*self.sigma)/(2*math.sqrt(self.M-self.t))) - (self.r * self.K * math.exp(-self.r * (self.M-self.t))*self.N(d2)) + (self.d*math.exp(-self.d*(self.M-self.t))*self.S0*self.N(d1))
        else:
            theta=((-math.exp(self.d*(self.M-self.t))*self.S0 * self.dN(-d1)*self.sigma)/(2*math.sqrt(self.M-self.t))) + (self.r * self.K * math.exp(-self.r * (self.M-self.t))*self.N(-d2)) - (self.d*math.exp(-self.d*(self.M-self.t))*self.S0*self.N(-d1))
        return theta

    def rho(self):
        d1 = self.d1f(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d)
        d2 = d1 - self.sigma * math.sqrt(self.M - self.t)
        if(self.CP == 'Call'):
            rho = self.K * (self.M - self.t) * math.exp(-self.r * (self.M - self.t)) * self.N(d2)
        else:
            rho = -self.K * (self.M - self.t) * math.exp(-self.r * (self.M - self.t)) * self.N(-d2)

        return rho
