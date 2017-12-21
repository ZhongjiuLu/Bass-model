#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:28:26 2017

@author: Zhongjiulu
"""

# import neccessary packages for Bass Model forecasting
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import leastsq # for OLS modelling
import math

# time vector
week = np.linspace(1, 12, num=12)
# revenue input
revenue = np.array([0.10,3.00,5.20,7.00,5.25,4.90, 3.00, 2.40, 1.90, 1.30,0.80,0.60])
# cummulative revenue input 
revenue_cum = np.cumsum(revenue)

# initial values of M, P, Q
values = [34.85, 0.074, 0.49]

# the realised values for prediction
week_4 = week[0:4]
revenue_4 = revenue[0:4]
revenue_4c = revenue_cum[0:4]
    
# residual function
def residual(values, week, rev, rev_cum):
    M = values[0]
    P = values[1]
    Q = values[2]
    bass = (P + Q * (rev_cum / M))*(M - rev_cum)
    return bass - rev

# perform prediction analsis for week 5 to week 12
for i in range(5, 13):
    var_final= leastsq(residual, values, args=(week_4, revenue_4, revenue_4c))[0]
    m, p, q = var_final
    bass_ft_i = (p+q*(revenue_cum[i-1]/m))*(m-revenue_cum[i-1])
    
    week_4 = list(week_4); week_4.append(i); week_4 = np.array(week_4)
    revenue_4 = list(revenue_4); revenue_4.append(bass_ft_i); revenue_4 = np.array(revenue_4)
    revenue_4c = np.cumsum(revenue_4)

# revenue vs prediction graph
plt.plot(week, revenue_4, color = "grey")
plt.plot(week, revenue, color = "blue")
plt.title('Revenues and Forecast')
plt.xlabel('Week')
plt.ylabel('Revenue')
plt.legend(['Forecast values', 'Observed Values'], loc = "upper right")
plt.show()

# Cumulative revenue graph
plt.plot(week, revenue_4c, week, revenue_cum)
plt.title('Forecast cummulative vs observed cummulative')
plt.xlabel('Week')
plt.ylabel('Cummulative Revenue')
plt.legend(['Forecast values', 'Observed Values'], loc = "upper left")
plt.show()
    
    
    