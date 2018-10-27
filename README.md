# AccretionDisk
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:15:20 2018

@author: jenniferharber
"""

import numpy as np
import matplotlib.pyplot as plt


R_in   = 5    # radius of central body
M_body = 1    # mass of central body 
R_out  = 25   # radius of disc
N      = 5  # number of annuli 
H      = 1    # disc thickness
M_0    = 1    # mass entering disk 
alpha  = 2000 # viscosity constant
Factor = 0.1  # m_dot << 1
Time   = 200  # time to run simulation

# Function "accretion_disk_array" to split the disk into N annuli 
# Inputs: Number of Annuli, inner radius, outer radius
# Output: Array R of the radii of tha Annuli

def accretion_disk_array(N, R_in, R_out):
    R = np.empty(N)
    for i in range(N):
        R[i] = (((R_out - R_in) / N) * i) + R_in   # linear progression
    return R


# Function "viscous_timescale" to calculate the viscous velecity at a particular radius 
# using the formula given in (enter ref)
# Inputs: Radius
# Output: viscous velecity at that radius

def viscous_timescale(rad):
    t_viscous = 2 * np.pi * rad**(3/2)/(((H/R_out)**2) * alpha) 
    return t_viscous  


# Function "delta_visc_time" to calculate the delta viscous velecity at a given annulus 
# Inputs: Annulus
# Output: Delta Timescale

def delta_visc_time(annulus):
    delta_t = viscous_timescale(annuli[annulus+1]) - viscous_timescale(annuli[annulus])
    return int(delta_t)


# Function "m_dot" to calculate variable rate of change of mass in a particular annulus 
# Inputs: None as it cycles through each annulus
# Output: Array of lowercase m_dot changes

def m_dot():
    m_dot = np.empty((len(annuli),Time))
    for annulus in range(N):
        # For each annulus calculate the radius and viscous timescale
        # then for each time division use the timescale as the frequency of a sine wave
        
        r=annuli[annulus]
        timescale = int(viscous_timescale(r) / (2.0 * np.pi)) #int to try and line up waves as timescale should be a
        for t in range(Time):
            m_dot[annulus][t] = Factor * np.sin(t / timescale)
    return m_dot
  

# Function "M_dot" to calculate the overall rate of change of mass in a particular annulus 
# Inputs: None as it cycles through each annulus
# Output: Array of Capital M_dot changes
 
def M_dot():
    M_dot = np.empty((len(annuli),Time))
    #For outer annulus there is no other input than the base rate of flow into the disk (M_0)and the small change in that annuli
    for t in range(Time):
        M_dot[(len(annuli)-1)][t] = M_0 * (1.0 + m_dot[(len(annuli)-1)][t])
    #For inner annului the rate of flow is determined by the rate from the next annuli and the small change m_dot   
    #the rate from the outer annuli is time delayed by the time it takes the mass to cross that annuli  (t_offset)
    #The code cycles through the annuli from the outside (penultimate annulus) to the inside
    #As the pattern is cyclical I assume that is the time is smaller than the offset we can start the cycle again
    for i in range(len(annuli)-2,-1,-1):
        t_offset_annulus = delta_visc_time(i)
        for t in range(Time):
            t_offset = t - t_offset_annulus
            if t_offset < 0:
                t_offset = t_offset + Time
            M_dot[i][t]=M_dot[i+1][t_offset] *(1.0 + m_dot[i][t]) 
    return M_dot

#Set up the disk with N annuli
annuli= accretion_disk_array(N, R_in, R_out)

#Calculate the small periodic changes in each annuli
m_dot=m_dot() 

# code to check / show m_dot across disk - shouldn't be needed later

#y_plot=np.empty(N*Time)
#x_plot=np.empty(N*Time)
#for i in range(N*Time):
#    x_plot[i]=R_in+(i * (R_out - R_in) / (N * Time))
#    y_plot[i]=m_dot[int(i/Time)][i % Time]
#plt.plot(x_plot,y_plot)
#plt.show()

#Calculate the flow in each annuli allowing for the input to that annuli and the small changes within it
M_dot=M_dot()


# code to check / show M_dot across disk - shouldn't be needed later

y_plot=np.empty(N*Time)
x_plot=np.empty(N*Time)
for i in range(N*Time):
    x_plot[i]=R_in+(i * (R_out - R_in) / (N * Time))
    y_plot[i]=M_dot[int(i/Time)][i % Time]
plt.plot(x_plot,y_plot)
plt.show()



