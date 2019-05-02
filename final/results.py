# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:30:06 2019

@author: jenniferharber
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
import numpy as np
#from random import random
from scipy.optimize import leastsq, fmin



break_frequency = np.array([-2.2,-2.2,-2.25,-2.7,-2.9,-3,-2.95,-3.4,-3.4,-3.4,-3.65,
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

def line_points( s, p, t ):
    return [ s * t[0] + p[0], s * t[1] + p[1], s * t[2] + p[2] ]

def weighted_dist( s, p, t, xVec, sigmaVec ):
    q = line_points( s, p, t )
    d  = ( q[0] - xVec[0] )**2 / sigmaVec[0]**2
    d += ( q[1] - xVec[1] )**2 / sigmaVec[1]**2
    d += ( q[2] - xVec[2] )**2 / sigmaVec[2]**2
    return np.sqrt( d )

def weighted_od( p, t, xVec, sigmaVec ):
    f = lambda s: weighted_dist( s, p, t, xVec, sigmaVec )
    sol = fmin( f, 0, disp=False )
    d = weighted_dist( sol[0], p, t, xVec, sigmaVec )
    return d

def residuals( params, data, sigmas ): ###data of type [ allx, ally, allz], sigma of type [allsx, allsy, allsz]
    px, py, pz, tx, ty, tz = params
    out = list()
    for x0, y0, z0, sx, sy, sz in zip( *( data + sigmas ) ):
        out += [weighted_od( [ py, py, pz ], [ tx, ty, tz ], [ x0, y0, z0 ], [ sx, sy, sz ] ) ]
    print (sum(out))
    return out

#myP = np.array( [ 1 , 1, 3 ] )
#myT = np.array( [ -1 ,-3, .8 ] )
#myT /= np.linalg.norm( myT )
#
sList = np.linspace( 0, 2, 100 )
#lineList = [ line_points( s, myP, myT ) for s in sList] 
#xData = [p[0] + .2 * ( 2 * random() - 1 ) for p in lineList ]
#yData = [p[1] + .4 * ( 2 * random() - 1 ) for p in lineList ]
#zData = [p[2] + .8 * ( 2 * random() - 1 ) for p in lineList ]

xyzData = [ mass, radius, break_frequency ]
sssData = [ len(mass) * [.2], len(mass) * [.4], len(mass) * [.8] ]

residuals( [ 0, 0, -2.5, 2, 2, -6 ],  xyzData, sssData )
myFit, err = leastsq(residuals, [0, 0, -2.5, 2, 2,-6 ], args=( xyzData, sssData ) )
print (myFit)

fitP = myFit[:3]
fitT = myFit[3:]
fitTN= np.linalg.norm( fitT )
fitT = [ fitT[0] / fitTN, fitT[1] / fitTN, fitT[2] / fitTN ]
fitLineList = [ line_points( s, fitP, fitT ) for s in sList ] 

ax = m3d.Axes3D(plt.figure() )
#ax.plot( *zip(*lineList) )
ax.plot( *zip(*fitLineList) )
ax.scatter3D( mass, radius, break_frequency )
plt.show()
