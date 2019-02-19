#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:03:37 2018

@author: jenniferharber
"""
# import modules 
import numpy as np
import matplotlib.pyplot as plt
import numbers
#import random
import datetime
from scipy.stats import cauchy
from scipy import signal
#from astroML.time_series.generate import generate_power_law

#----------------------------------------------------------------------------
'>>-------------------------------To-Do-List--------------------------------<<'
"""
change to unit Rg (distance) and Rt (time)
"""
#----------------------------------------------------------------------------
'>>--------------------------------Constants--------------------------------<<'

c     = 2.99792458e8    # Speed of Light [ metres per second ]
G     = 6.67259e-11     # Gravitational Constant  [ MKS units ] 
sigma = 5.6703e-8       # Stefan-Boltzmann Constant [Watt m^-2 K^-4]
k     = 1.38064852e-23  # Boltzman constant
h     = 6.62607004e-34  # Planks Constant
SM    = 1.99e30         # 1 solar mass (kg)
years = 3600*24*365.25  # seconds in a year 
pc    = 3.086e16        # parsecs in metres

#----------------------------------------------------------------------------
'>>--------------------------------Parameters-------------------------------<<'

R_in         = 5.0  # Radius of Central Body, eg. Black Hole, also the start of the disk  (Rg)
M_body = 20.0 * SM  # Mass of Central Body in solar masses (kg)
R_out        = 6000.0 # Radial distance to edge of disk (Rg)
N            = 1000   # Number of Annuli , N > 1 at all times 
M0_dot = 0.01 * SM / years # Accretion Rate at outer radii, where mass first flows in (kg/s) 
alpha        = 1.0  # Viscosity Parameter 
H_over_R     = 0.01 # ratio of H to R
scale_factor = 0.1  # m_dot << 1
red_noise_factor = 1e8 # factor to scale up red_noise
sample_ratio = 1 # this parameter sets the time resolution of the time-series for each annulus
                    # for each annulus, our time step is the local viscous_timescale / sample_ratio
cycles       = 10   # number of viscous_timescales of the outermost (slowest) annulus to span
                    # i.e. all annuli will have time-series spanning cycles*viscous_timescale(outer annulus)
sinusidal    = False
red_noise    = not(sinusidal)
Q            = 2.0  # width of lorentzian = 2 * F_visc / Q
incidence    = 0    # angle between line of sight and normal to disk plane
D            = 1000 * pc # Distance of observer from the disk
checks       = True # Check inputs to functions eg R_in >= r >= R_out 

'>>--------------------------------Units------------------------------------<<'

Rg   = (G * M_body) / (c * c)  # Unit of Distance (Arevalo & Uttley)
Rt   = Rg / c                  # Unit of Time     (Arevalo & Uttley)

D    = D / Rg                  # put distance to observer into these units

#----------------------------------------------------------------------------
'>>-------------------------------Conventions-------------------------------<<'
"""
  - r is always a single radius in Rg units
  - t is always a single time in Rt units
  - t_pos is always the single time position in a variable array of Mdot/mdot - 
 can be converted to t by multiplying by the time interval in that annulus
  - Where t and t_pos are inputs to a function t will be used unless it is set to -1
  - annuli is the array of the inner radii of the annuli the disk is split into.  
  - Annulus is always a particular ring in the array of annuli 
  - times is always an array of the number of time divisions per annulus  
    (how many divisions the annulus is split into)
  - interval is always an array of the time difference between these intervals
       (delta t)
  - start is an array of the start location for each annulus
  - i is used as a 'counter' in loops
"""
'>>--------------------------------Functions--------------------------------<<'


def accretion_disk (N, R_in, R_out):
    """
    Creates an array of all the annuli (defined by calling accretion disk 
    array so annuli = R)
    Function "accretion_disk" to split the disk into N annuli   
    Annuli are geometrically distributed so that r/(r+1) is a constant 

    Inputs: Number of annuli, inner radius and outer radius of the accretion disk.
    Output: Array R of the inner radii of the annuli in the disk
    """
    
    if checks:     # sanity check of input to function
        if R_in >= R_out or N < 10:
            print ('ERROR in input to function:accretion_disk')
            
    # create an empty array:     
    annuli = np.empty(N)
    
    # determine the geometric ratio
    ratio = (R_out / R_in) ** (1.0/(N - 1))
    
    # loop through the annuli setting the radius for each
    for i in range(N):
        #R[i] = (((R_out - R_in) / N) * i) + R_in   # linear progression
        
        # annuli[i] is the inner radius of the i th annulus
        
        annuli[i] = R_in * (ratio ** i)                  # geometric progression
        
    return annuli


def viscous_timescale(r):    
    """
    Function "viscous_timescale" to calculate the viscous timescale at a particular radius 
    Using the formula given in Arevelo and Uttely
    alpha is the viscosity parameter which depends on the type of disk being measured
    
    
    Inputs: radius (in units of Rg)
    Output: viscous timescale at that radius (in units of Rt)
    """
    
    t_viscous = (2 * np.pi) * r ** (3.0 / 2.0) / ((H_over_R ** 2.0) * alpha)
    
    return t_viscous  


def phase_shift(annuli,annulus):
    """
    Function "Phase_Shift" to calculate the delta viscous velecity at a given
    annulus from the next outer one.
    Inputs: annulus and the annuli array
    Output: Delta Timescale
    """
    
    if checks:     # sanity check of input to function
        if annulus > len(annuli) or annulus < 0:
            print ('ERROR in input to function:phase_shift. Annulus must be within the array',annulus)
    
    # delta_t is the difference in time taken to get from an annuli to the centre
    # for two consecutive annuli and is therefore the time taken to get from the 
    # outer annuli to next annuli (delta_t is local to this function)         
    delta_t = viscous_timescale(annuli[annulus+1]) - viscous_timescale(annuli[annulus])
    return delta_t



def time_division(annuli):
    """
    This calculates the number of time divisions required for each annulus
    sample ratio * viscous timescale for that annulus
    input: array of annuli radii
    outputs: three arrays (same length as the annuli array);
    
        time_div = the number of time steps required to cover cycles * 
                viscous_timescale(outer annulus)

        time_interval = the corresponding time step for the local annulus
    
        annulus start = the start location of each annulus with a variable array (sum of
                time divisions within that annulus)
    
    
    """
    # radius of the outermost annulus in Rg
    rmax = annuli[len(annuli)-1]
    
    
    total_time = cycles * viscous_timescale(rmax)
    # this is the total simulation time;  cycles * the viscous timescale of the
    # outermost (hence, slowest) annulus. Therefore same for every annulus. 
    
    # Create empty arrays (number of divisions and start locations are integer values)
    # time divisions - number of time steps in each annulus. 
    time_div = np.empty(len(annuli),dtype=int) 
    
    # annulus start position -  used to locate each annulus in arrays of mdot/Mdot
    annulus_start = np.empty(len(annuli),dtype=int)
    
    # time intervals - difference in time between each time step different for 
    # each annulus. Measured in Rt. 
    time_interval = np.empty(len(annuli))
    
    #initialize both time series arrays and keep a count of the start of each annuli
    count = 0
    for annulus in range(len(annuli)):
        r = annuli[annulus]       #this is the radius of a given annulus in Rg
        
        annulus_start[annulus] = count
        
        # time interval set to allow more entries where the timescale is longer. 
        time_interval[annulus] = viscous_timescale(r) / sample_ratio 

        time_div[annulus] = int(total_time/time_interval[annulus])
        
        # update the running total to keep track of where the next annulus starts. 
        count = count + time_div[annulus]
        
    return time_div, time_interval, annulus_start


def create_variable_array(times):
    """
    This creates a 1D array with the time readings for annulus 0 followed by 1 
    and then 2 and so on. The number of entries is the sum of the time divisions
    per annulus. 
    Input: array of divisions
    Output: an empty array to hold all of these divisions
    """
    
    R = np.empty(np.sum(times))
    return R


def radius_to_annulus(r,annuli):    
    """
    Returns the annulus in which a certain radius can be found. If the radius is
    larger than the disk then the outer annulus is returned
    
    Input: radius and array of annuli
    Output: Annulus which contains the radius
    """
    
    if checks:     # sanity check of input to function
        if r < R_in:
            print ('ERROR in input to function:radius to annulus. radius can not be smaller than R_in')
            return -1  
          
    for annulus in range(len(annuli)):
        if annuli[annulus] == r:
            annulus_smaller = annulus
            return annulus_smaller    
        if annuli[annulus] > r:
            annulus_smaller = annulus-1
            return annulus_smaller
        
    if checks:     # sanity check of input to function
        print ('Warning in function:radius to annulus. radius larger than R_out - set to R_out')
    return len(annuli)-1

    
def read_variable_array(array,annuli,times,interval,start,t,t_pos,r):
    """
    generalised function to deal with treating the 1D array given specific 
    radius and time inputs. 
    if t is given then it is used to to give an linearly interpolated value between
    the two nearest timestamps.
    if t is set to -1 the t_pos position is used to give the value at that position.
    Inputs: variable array to read, annuli, divisions, time intervals, start locations,
    time or time position (time takes precedence) and radius
    Output: value of the given variable array at the radius and time specified    
    """
    
    # determine the annulus from the radius and find its start location
    annulus = radius_to_annulus(r,annuli)    
    annulus_start = start[annulus]
    
    #check to determine is an exact position has been specified, if not reset t_pos based on t
    if t != -1:
        # t has been specified so calculate t_pos from it. 
        t_pos = int(t / interval[annulus])
    
    if checks:     # sanity check of input to function
        if r > np.max(annuli) or r < np.min(annuli):
            print ('ERROR in input to function:read_variable_array. r out of range',r,annulus)
        if t_pos >= times[annulus]:
            print ('ERROR in input to function:read_variable_array. t greater than allowed in annuli',t,t_pos,annulus)
    
    #check to determine is an exact position has been specified, if not interpolate value        
    if t != -1: 
        
        #interpolated result:
        value1 = array[annulus_start + t_pos]               # value before
        
        # check time position still in current annulus
        if t_pos + 1 < times[annulus]:                     
            value2 = array[annulus_start + t_pos + 1]       # value after
        else:
            value2 = array[annulus_start]                    # value after if looping around 
        # interpolate value from equation for linear interpolation. 
        value = value1 + ((value2 - value1) * (t  - (t_pos * interval[annulus])))/ interval[annulus]
    else:
        # if t_pos is known exactly then use the value. 
        value = array[annulus_start + t_pos]
        
    return value        


def update_variable_array(array,annuli,times,interval,start,t,t_pos,r,value):    
    """
    generalised function to deal with updating the 1D array given radius and time inputs
    if t is given then it is used to to update the value before that time
    if t is set to -1 the t_pos position is used to update the value at that position
    Inputs: variable array to read, annuli, divisions, time intervals, start locations,
    time or time position (time takes precedence), radius and value to be placed in array
    Output: none 
    """
    
    #determine the annulus from the radius (needed to get the start of annulus position)
    annulus = radius_to_annulus(r,annuli)
    
    #check to determine is an exact position has been specified, if not reset t_pos based on t
    if t != -1:
        # t has been specified so calculate t_pos from it. 
        t_pos = int(t / interval[annulus])

    if checks:     # sanity check of input to function
        if r > np.max(annuli) or r < np.min(annuli):
            print ('ERROR in input to function:update_variable_array. r out of range',r,annulus)    
        if t_pos > times[annulus]:
            print ('ERROR in input to function:update_variable_array. t greater than allowed in annuli',t,t_pos,annulus)
    
    # Determine the start position of the annulus        
    annulus_start = start[annulus]

    # set the value
    array[annulus_start + t_pos] = value
    return ()
    

def m_dot(annuli,times,interval,start):
    """
    Function: "m_dot" to calculate variable rate of change of mass in a particular annulus 
        For each annulus calculate the radius and viscous timescalethen 
        for each time division use the timescale as the frequency of a sine wave
        calculate the small periodic changes in each annuli
        
    Inputs: array of annuli,array of annuli,array of time divisions (how many per annulus),
    array of start locations for each annulus in the variable array
    Output: Array of lowercase m_dot changes
    """
    
    # Create the variable array of m dot at each annulus/time position 
    m_dot = create_variable_array(times)
    
    # Choice of variability in m_dot sinusidal or red noise (power law/lorentzian)
    if sinusidal:
        for annulus in range(len(annuli)):
            # For each annulus calculate the radius and viscous timescale
            # then for each time division use the timescale as the frequency of a sine wave
            r = annuli[annulus]
            timescale = viscous_timescale(r) / (2.0 * np.pi) 
            for t_pos in range(times[annulus]):
                t = t_pos * interval[annulus]
                value = scale_factor * np.sin( t / timescale)
                update_variable_array(m_dot,annuli,times,interval,start,-1,t_pos,r,value)
    if red_noise:
        for annulus in range(len(annuli)):
            # For each annulus calculate the radius and use the lorentzian 
            # function to return the values for time divisions within that annulus. 
            
            r = annuli[annulus]
            # use function to return array of values 
            for i in range(cycles):
                values = red_noise_factor * generate_lorenztian_law(int(times[annulus]/cycles), interval[annulus], Q,(1.0 / viscous_timescale(r)))
            
            # for each t value update the variable array
                for j in range(int(times[annulus]/cycles)):
                    t_pos = j + i * int(times[annulus]/cycles)
                    update_variable_array(m_dot,annuli,times,interval,start,-1,t_pos,r,values[j])
                
    return m_dot


def M_dot(m_dot,annuli,times,interval,start):
    """
    Function: "M_dot" to calculate the overall rate of change of mass in a particular 
    annulus. For inner annului the rate of flow is determined by the rate from the next
    outer annuli and the small change m_dot. the rate from the outer annuli is time delayed by
    the time it takes the mass to cross that annuli  (t_offset).The code cycles through
    the annuli from the outside (penultimate annulus) to the inside. As the pattern is 
    cyclical I assume that if the time is smaller than the offset we can start the cycle 
    again.
    
    Inputs: m_dot (small changes in annulus), annuli and times,
            array of start locations for each annulus in the variable array
    Output: Array of Capital M_dot changes
    """
    #create the empty array
    M_dot = create_variable_array(times)
    
    #For outer annulus there is no other input than the base rate of flow into
    #the disk (M0_dot) and the small change (m_dot) in that annuli 
    
    # Determine the radius of the outer annulus
    r = annuli[len(annuli)-1]
    
    # For each time division in the array calculate M_dot and update the variable array
    for t_pos in range(times[len(annuli)-1]):
        
        value = M0_dot * (1.0 + read_variable_array(m_dot,annuli,times,interval,start,-1,t_pos,r))
        update_variable_array(M_dot,annuli,times,interval,start,-1,t_pos,r,value)
        
    # For inner annuli the base rate of flow into
    # the annulus is the M_dot from the next outer annulus (offset by the 
    # time it takes to cross the annulus - phase_shift)
   
   
    for annulus in range(len(annuli)-2,-1,-1): # work backwards through array
        r  = annuli[annulus]    # this annulus
        r1 = annuli[annulus+1]  # next outer annulus (from whence M_dot arrives)
        
        # Determine the time offset from the outer annulus
        t_offset_annulus = phase_shift(annuli,annulus)
        
        # For each time division calculate M_dot and update the variable array
        for t_pos in range(times[annulus]):
            
            # Change time position to proper time (Rt)
            t_offset = (t_pos * interval[annulus]) - t_offset_annulus 
            if t_offset < 0:    # If subtracting the offset makes the time negative loop round the timescale
                t_offset = t_offset + ((times[annulus]-1) * interval[annulus])
                                          
            value = read_variable_array(M_dot,annuli,times,interval,start,t_offset,-1,r1
                                        ) * (1.0 + read_variable_array(m_dot,annuli,times,interval,start,-1,t_pos,r))
            update_variable_array(M_dot,annuli,times,interval,start,-1,t_pos,r,value)
    return M_dot


def effective_temperature(r,t_pos,t,M_dot,annuli,times,interval,start): 
    """
    Effective Temperature at a particular radius and time position 
    Inputs: radius, time, time position,M_dot (total mass changes in annulus), annuli, 
    time divisions,time intervals and array of start locations for each annulus in the variable array
    Output: Local effective temperature (in Kelvin)
    """
    
    # check to determine is an exact position has been specified, if not reset t_pos based on t
    if t_pos != -1:
        # t has been specified so calculate t_pos from it. 
        # Determine M_dot at the specified radius and time. Use this to calculate temperature
        M_dot_value = read_variable_array(M_dot,annuli,times,interval,start,-1,t_pos,r)
    else:
        #t_pos = -1
        M_dot_value = read_variable_array(M_dot,annuli,times,interval,start,t,-1,r)
        
    # Determine M_dot at the specified radius and time. Use this to calculate temperature
    #M_dot_value = read_variable_array(M_dot,annuli,times,interval,start,-1,t_pos,r)
    T_eff = (3 * G * M_body * M_dot_value * (1 - ((R_in / r) ** 0.5)) / (
            (r ** 3) * 8 * np.pi * sigma)) ** 0.25
            
    # return the effective temperature adjusted for the units of Rg        
    return T_eff / (Rg ** 0.75)


def annulus_effective_temperature(M_dot,annuli,times,interval,start):
    """
    Effective Temperature, integrated across time for an annulus
    used in determining the blackbody spectrum. repeated for each annulus to 
    give the temperature across the disk
    Inputs: M_dot (total mass changes in annulus), annuli, 
    time divisions,time intervals and array of start locations for each annulus in the variable array
    Output:  array of effective temperatures (in Kelvin) per annulus
    """
    
    # Create an empty array to hold the values
    annuli_temperature = np.empty(len(annuli))
    
    # loop through the annuli to get a result for each annulus and add it to the array
    for annulus in range(len(annuli)):
        
        # find the radius of the annulus and reset the counter to zero for each annulus
        r=annuli[annulus]
        total_effective_temp = 0
        
        # Loop through the time divisions calulating each temperature and adding it to the counter (total effective temp)
        for t_pos in range(times[annulus]):
            total_effective_temp = total_effective_temp + effective_temperature(
                    r,t_pos,-1,M_dot,annuli,times,interval,start)
            
        # Divide the total temperature by the number of readings to get the average    
        average_effective_temp = total_effective_temp / times[annulus]
        
        # Update the array
        annuli_temperature[annulus] = average_effective_temp
        
    # Return the finished array of temperatures by annulus    
    return annuli_temperature

    
def flux_at_frequency(annuli,frequency,M_dot,times,interval,temperature):
    """
    Determine the flux for a specified frequency by integrating the BB radiation
    across the disk
    Inputs: annuli, the frequecncy to be used for the calculation, M_dot (total mass changes in annulus),  
    time divisions,time intervals and array of temperatures for each annulus 
    Output:  Flux from the disk at the specified frequency
    """
    
    # set the total to zero before adding components from each annulus
    total_integral = 0
    
    
    # Loop thgrough thye annuli adding the flux from each to the total
    for annulus in range(len(annuli)):
        
        # integrating using dR (annulus thinkness)
        # dR is the difference between radii of adjacent annuli unless it is the
        # outer annulus which doesnt have an adjacent one, in this case we use 
        # the total disk radius as the outer radius. 
        if annulus != len(annuli) - 1:
            dR = annuli[annulus+1] - annuli[annulus]
        else:
            dR = R_out - annuli[annulus]
            
        # For each annulus determine the radius and effective temperature
        r = annuli[annulus]
        T_eff = temperature[annulus]
        
        # Find the area under the curve for this annulus and add it to the total
        area = dR * r / (np.exp(h*frequency/(k * T_eff)) - 1)
        total_integral = total_integral + area
        
    # adjust the flus with the required constants from the equation and return it
    # units = watts/m^2
    
    flux = 4 * np.pi * np.cos(incidence) * (frequency ** 3) * total_integral * h / ((c * D) ** 2)
    return flux     
       
  
def instantaneous_flux_at_frequency(annuli,frequency,M_dot,times,interval,start,t):
    """
    Determine the flux for a specified frequency by integrating the BB radiation
    across the disk
    Inputs: annuli, the frequecncy to be used for the calculation, M_dot (total mass changes in annulus),  
    time divisions,time intervals and array of temperatures for each annulus 
    Output:  Flux from the disk at the specified frequency
    """
    
    # set the total to zero before adding components from each annulus
    total_integral = 0
    
    
    # Loop thgrough thye annuli adding the flux from each to the total
    for annulus in range(len(annuli)):
        
        # integrating using dR (annulus thinkness)
        # dR is the difference between radii of adjacent annuli unless it is the
        # outer annulus which doesnt have an adjacent one, in this case we use 
        # the total disk radius as the outer radius. 
        if annulus != len(annuli) - 1:
            dR = annuli[annulus+1] - annuli[annulus]
        else:
            dR = R_out - annuli[annulus]
            
        # For each annulus determine the radius and effective temperature
        r = annuli[annulus]
        T_eff = effective_temperature(r,-1,t,M_dot,annuli,times,interval,start)
        
        # Find the area under the curve for this annulus and add it to the total
        area = dR * r / (np.exp(h*frequency/(k * T_eff)) - 1.0)
        total_integral = total_integral + area
        
    # adjust the flus with the required constants from the equation and return it
    # units = watts/m^2
    
    flux = 4 * np.pi * np.cos(incidence) * (frequency ** 3.0) * total_integral * h / ((c * D) ** 2.0)
    return flux 


def generate_lorenztian_law(N, dt, beta, f_visc, generate_complex=False, random_state=None):
    
    """Generate a lorentzian-law light curve

    This uses the method from Timmer & Koenig [1]_

    Parameters
    ----------
    N : integer
        Number of equal-spaced time steps to generate
    dt : float
        Spacing between time-steps
    beta : float
        Ratio of width to the viscous frequency. 
    f_visc : float
             viscosity frequency - centre and base for ratio of width. 
    generate_complex : boolean (optional)
        if True, generate a complex time series rather than a real time series
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator

    Returns
    -------
    x : ndarray
        the length-N

    References
    ----------
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)

    Npos = int(N / 2)
    domega = (2 * np.pi / (dt * N))

    #create lorenztian distribution (same as cauchy)
    # parameters are centre and width. 
    dist = cauchy(f_visc, f_visc / beta)
     
    if generate_complex:
        omega = domega * np.fft.ifftshift(np.arange(N) - int(N / 2))
    else:
        omega = domega * np.arange(Npos + 1)

    x_fft = np.zeros(len(omega), dtype=complex)
    
    #create gaussian distribution for real and imaginary parts
    x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
    x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

#    x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
#    x_fft[1:] *= (1. / np.sqrt(2))

    for i in range(len(omega)):
        # white noise (gaussian) multiplied by filter function (lorentzian)
       x_fft.real[i] = dist.pdf(i) * x_fft.real[i]
       x_fft.imag[i] = dist.pdf(i) * x_fft.imag[i]
#       x_fft[i] = lambda_visc2 / (x_fft[i] - angular_visc2 + lambda_visc2)

    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, N)

    return x



def check_random_state(seed):
    """
    Turn seed into a `numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : `None`, int, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.

    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.

    Notes
    -----
    This routine is from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


'>>--------------------------------Main Code--------------------------------<<'

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Annuli Number=",N,"cycles=",cycles,"samples=",sample_ratio)

#these are the radii corresponding to the disk annuli in units of Rg
disk_annuli = accretion_disk(N, R_in, R_out)

# set up the time divisions and intervals and also the start locations of annuli 
# for the variable arrays. 
disk_time_divisions, disk_time_interval, disk_annulus_start = time_division(disk_annuli)

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: disk (radius,intervals, divisions and start locations) set up")

# plotting the radius of the disk against time divisions 
# this is only a test to check code plot isnt required. 
plt.plot(disk_annuli, disk_time_divisions)
plt.xlabel("Disk Radius", fontsize =10)
plt.ylabel("Time Divisions", fontsize =10)
plt.savefig ('time_divisons.pdf')
plt.show()

# set up variable array of m dot for every time position for every annulus
# (small variable mass movement wihin an annulus.)
disk_m_dot = m_dot(disk_annuli,disk_time_divisions,disk_time_interval,disk_annulus_start) 

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: m_dot set up")

# plotting the m dot against time for three chosen annuli.  
# this is only a test to check code, plots isnt required. 
for annulus in [1,int(N/2),N-2]:
    #annulus=random.randint(0,N-1)
    # create array of mdot for a single annulus by taking the start of the annulus
    # until the start of the next annulus and then back one. 
    m_dot=disk_m_dot[disk_annulus_start[annulus]:disk_annulus_start[annulus+1]-1]
    # actual time (Rt)
    time=range(1,disk_time_divisions[annulus])*disk_time_interval[annulus]
    plt.plot(time, m_dot,label=annulus)
    plt.xlabel("Time", fontsize =10)
    plt.xticks([])
    plt.yticks([])
    ylabel = "m_dot at Annulus" + str(annulus)
    file=ylabel + ".pdf"
    plt.ylabel(ylabel, fontsize =10)
    plt.savefig (file)
    plt.show()
    
    
    # plot the fourier transform to check that the lorentzian power law code 
    # has worked. 
    fft_m_dot=np.fft.rfft(m_dot)/len(m_dot) # normalised fourier transform.
    # fourier transform gives twice the number of points so take first half. 
    fft_m_dot=fft_m_dot[range(int(len(m_dot)/2))]
    # reduce time frame to match useful points from the fourier transform. 
    time=time[range(int(len(m_dot)/2))]
    # square:
    fft_m_dot2 = fft_m_dot ** 2
    #plt.plot(time, fft_m_dot,label=annulus)
    plt.plot(time,1/ fft_m_dot2,label=annulus)
    plt.xlim(0, 2E7)
    plt.xlabel("Status: time", fontsize =10)
    plt.ylabel("Status: rfft_m_dot^2", fontsize =10)
    plt.show()



# set up variable array of M dot for every time position for every annulus
# (overall mass movement wihin an annulus.)
disk_M_dot = M_dot(disk_m_dot,disk_annuli,disk_time_divisions,disk_time_interval,disk_annulus_start) 

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: M_dot set up")

# plotting the m dot against time for three chosen annuli.  
# this is only a test to check code, plots isnt required. 
#for plots in range(3):
for annulus in [1,int(N/2),N-2]:
    #annulus=random.randint(0,N-1)
    # create array of Mdot for a single annulus by taking the start of the annulus
    # until the start of the next annulus and then back one. 
    M_dot=disk_M_dot[disk_annulus_start[annulus]:disk_annulus_start[annulus+1]-1]
    # actual time (Rt)
    time=range(1,disk_time_divisions[annulus])*disk_time_interval[annulus]
    plt.plot(time, M_dot,label=annulus)
    plt.xlabel("Time", fontsize =10)
    #plt.xticks([])
    #plt.yticks([])
    ylabel = "M_dot at Annulus" + str(annulus)
    file=ylabel + ".pdf"
    plt.ylabel(ylabel, fontsize =10)
    plt.savefig (file)
    plt.show()

# set up array of effective temperature (K) for every annulus
disk_temperature = annulus_effective_temperature(disk_M_dot,disk_annuli,disk_time_divisions,disk_time_interval,disk_annulus_start) 

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: Temperature set up")

 # plot radius against temperature (not needed used as a check)
plt.plot(disk_annuli, disk_temperature)
#plt.xticks([])
#plt.yticks([])
plt.xlabel("Disk Radius", fontsize =10)
plt.ylabel("Temperature", fontsize =10)
plt.savefig ('temperature.pdf')
plt.show()


# using all the data modelled on the disk above (temperature, radius, Mdot, mdot etc)
# we can now start using the model to make prediction eg flux from the disk. 
num     = 1000     # number of points to plot chosen to give a smooth graph
y1_plot = np.empty(num) # set up empty array of y coordinates
x_plot  = np.empty(num) # set up empty array of x coordinates

# for each point (num) calculate a frequency and use that frequency to calculate 
# the flux given by the disk. 
# loop through num records using i as the counter. 
for i in range(num):
    # calculate frequency (expecting x rays)
    freq = (i)*1e17  
# commented code - log version:
 #   x_plot[i]  = np.log(freq)
 #   y1_plot[i] = np.log(flux_at_frequency(disk_annuli, freq, disk_M_dot, disk_time_divisions, disk_temperature))
    
    x_plot[i]  = (freq)
    y1_plot[i] = (flux_at_frequency(disk_annuli, freq, disk_M_dot, disk_time_divisions, disk_time_interval, disk_temperature))

plt.xlabel("frequency (Hz)", fontsize =10)
plt.ylabel("flux (Wm^-2) ", fontsize =10)
plt.plot(x_plot,y1_plot,'r',linewidth=1, label ='$Cˆ{12}$' )

#plt.xticks([])
plt.yticks([])  #turn off axis numbers on y axisto make graph clearer
plt.savefig ('spectum.pdf')
plt.show()

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status:Spectrum produced")

record_time     = 100 #time over which counts are taken
inner_divisions = int(disk_time_divisions[0]-1) #maximum amount of divisions
time_series     = np.empty(int(inner_divisions / record_time)) #set up x value array
counts          = np.empty(int(inner_divisions / record_time)) #set up y value array
frequency       = 0.6e20 #x-ray

for i in range(int(inner_divisions/record_time)): #loop through number of measurements
    observation_count = 0 #initialise observation count for each measurement
    for j in range(record_time): #loop through the time over which counts are taken 
        t = ((i * record_time) + j) * disk_time_interval[0] #convert tpos to time value
        observation_count = observation_count + instantaneous_flux_at_frequency(
                disk_annuli,frequency,disk_M_dot,disk_time_divisions,disk_time_interval,disk_annulus_start,t)
        #observation count is accumulated by adding the flux at each time value. 
    counts[i] = observation_count / (h * frequency) #convert from energy to number of photons
    time_series[i]=t #set time of measurement to be end of measurement.
    
#plot time against counts which gives a basic lightcurve   
plt.xlabel("time", fontsize =10)
plt.ylabel("counts ", fontsize =10)
plt.plot(time_series,counts,'r',linewidth=1, label ='$Cˆ{12}$' )

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: Lightcurve Produced")

#plt.xticks([])
#plt.yticks([])
plt.savefig ('lightcurve.pdf')
plt.show()

#use mathplot.psd to give a power spectrum density and return pds and frequency arrays
pds, freqs =  plt.psd(counts, 512, 1 / record_time) 
plt.savefig ('plot.psd.pdf')
plt.show()

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: Basic PDS produced")

#replot using the log of the frequency
log_freq = np.log10(freqs)
plt.xlabel("Log Frequency", fontsize =10)
plt.ylabel("PSD ", fontsize =10)
plt.plot(log_freq,pds,'r',linewidth=1, label ='$Cˆ{12}$' )
plt.savefig ('plot.logpsd.pdf')
plt.show()

#replot also using log of pds
log_pds = np.log10(pds)
plt.xlabel("Log Frequency", fontsize =10)
plt.ylabel("Log PSD ", fontsize =10)
plt.plot(log_freq,log_pds,'r',linewidth=1, label ='$Cˆ{12}$' )
plt.savefig ('plot.logfreq.logpsd.pdf')
plt.show()

# print parameters and time stamp so we know what is running and how long it takes. 
now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"), "Status: Log PDS's Produced")

#use periodogram method and return frequency and power density:
f, Pxx_den = signal.periodogram(counts,record_time)
plt.semilogy(np.log10(f), Pxx_den) #power density already log
#plt.ylim([1e-7, 1e2])
plt.xlabel('log frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.savefig ('logsemilogy.pdf')
plt.show()

# print time so we know how long the code ran for (e.g if left over night)
now = datetime.datetime.now()
#print (now.strftime("%Y-%m-%d %H:%M"))
print (now.strftime("%Y-%m-%d %H:%M:%S"),len(disk_M_dot))