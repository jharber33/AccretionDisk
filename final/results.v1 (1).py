# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:30:06 2019

@author: jenniferharber
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
#from random import random
#from scipy.optimize import leastsq, fmin



frequency = np.array([-2.2,-2.2,-2.25,-2.7,-2.9,-3,-2.95,-3.4,-3.4,-3.4,-3.65,
                            -2.4,-2.5,-2.6,-3,-3.1,-3.1,-3.2,-3.35,-3.6,-3.7,-3.55,
                            -2.5,-2.6,-2.75,-3.3,-3.3,-3.2,-3.3,-3.65,-3.8,-3.65,-4.1,
                            -2.9,-2.9,-3,-3.4,-3.5,-3.45,-3.55,-3.9,-4,-3.85,-4,
                            -3,-3.05,-3.15,-3.55,-3.6,-3.65,-3.6,-4,-4.2,-4.1,-4.2,
                            -3.15,-3.1,-3.15,-3.55,-3.8,-3.8,-3.95,-4.1,-4.25,-4.3,-4.25,
                            -3.4,-3.3,-3.45,-3.85,-4,-4,-4.15,-4.2,-4.6,-4.45,-4.3,
                            -3.65,-3.6,-3.6,-3.9,-4.1,-4.2,-4.2,-4.25,-4.7,-4.5,-4.6,
                            -3.7,-3.7,-3.85,-4.2,-4.3,-4.3,-4.4,-4.35,-4.8,-4.75,-4.8,
                            -3.8,-3.8,-3.9,-4.3,-4.4,-4.55,-4.55,-4.5,-4.9,-5.1,-5.35,
                            -4.05,-4.05,-4.2,-4.5,-4.6,-4.65,-4.6,-4.8,-4.9,-5.1,-5.3])



mass=             np.array([0,0,0,0,0,0,0,0,0,0,0,
                 0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
                 0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
                 0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                 0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,
                 1,1,1,1,1,1,1,1,1,1,1,
                 1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,
                 1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,
                 1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,
                 1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,
                 2,2,2,2,2,2,2,2,2,2,2]
                 )

radius=             np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,
                  1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])

radius1d = np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
mass1d = np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
mass1d, radius1d = np.meshgrid(mass1d, radius1d)
frequency2d = np.empty((11,11))
for m in range(11):
    for r in range(11):
        frequency2d[m,r]=frequency[(m*11)+r]
        
        
    

# plot raw data
fig=plt.figure()
ax = plt.subplot(111, projection='3d')



ax.view_init(30, 00)
surf=ax.plot_surface( mass1d, radius1d, frequency2d, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)



## do fit
#tmp_A = []
#tmp_b = []
#for i in range(len(mass)):
#    tmp_A.append([mass[i], radius[i], 1])
#    tmp_b.append(frequency[i])
#b = np.matrix(tmp_b).T
#A = np.matrix(tmp_A)
#fit = (A.T * A).I * A.T * b
#errors = b - A * fit
#residual = np.linalg.norm(errors)
#
#print ("solution:")
#print ("frequency = %f Mass + %f radius + %f" % (fit[0], fit[1], fit[2]))
#print ("errors:")
#print (errors)
#print ("residual:")
#print (residual)
#
## plot plane
#xlim = ax.get_xlim()
#ylim = ax.get_ylim()
#X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
#                  np.arange(ylim[0], ylim[1]))
#Z = np.zeros(X.shape)
#for r in range(X.shape[0]):
#    for c in range(X.shape[1]):
#        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
#ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('Log_Mass(M_sol)')
ax.set_ylabel('Log_radius(Rg)')
ax.set_zlabel('Log_Frequency(Hz)')
plt.savefig ("slope1.pdf")
plt.show()
