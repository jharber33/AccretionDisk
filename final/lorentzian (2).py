# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:26:39 2019

@author: Harbers
"""

import numpy as np
from scipy.stats import cauchy
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# Define the distribution parameters to be plotted
Q=2
freq = 100
gamma = freq/(2*Q)
#linestyles = '-'
mu = freq
x = np.linspace(0, 200, 1000)

#------------------------------------------------------------
# plot the distributions
#fig, ax = plt.subplots(figsize=(5, 3.75))

dist = cauchy(mu, gamma)

plt.plot(x, dist.pdf(x), color='k' , label="mu=100, gamma=25")

z=dist.pdf(x)
y1=np.linspace(0, max(z)-.0001, 50)
x1 = y1 * freq / y1
#x1 = freq
plt.plot(x1, y1, color='r')



x2=np.linspace(51, 149, 50)
y2= x2 * 2.550562029164567604e-03 / x2
plt.plot(x2, y2, color='b')
#
#plt.xlim(0, 200)
#plt.ylim(0, 0.65)
#
plt.xlabel("frequency")
plt.ylabel("probability")
#plt.title('Cauchy Distribution')
#
plt.legend()
plt.savefig ("lorentzian.pdf")
plt.show()