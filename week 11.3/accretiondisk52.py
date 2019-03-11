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
import datetime
from scipy.stats import cauchy
from scipy import signal
import gc

#----------------------------------------------------------------------------
'>>-------------------------------To-Do-List--------------------------------<<'
"""

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
M_body       = 20.0 * SM  # Mass of Central Body in solar masses (kg)
R_out        = 10000.0 # Radial distance to edge of disk (Rg)
number_of_annuli = 200  # Number of Annuli , N > 1 at all times 
M0_dot       = 0.01 * SM / years # Accretion Rate at outer radii, where mass first flows in (kg/s) 
alpha        = 1.0  # Viscosity Parameter 
H_over_R     = 0.01 # ratio of H to R
scale_factor = 0.1  # m_dot << 1
red_noise_factor = 5e-10 # factor to scale up red_noise
sample_ratio = 8   # this parameter sets the time resolution of the time-series for each annulus
                    # for each annulus, our time step is the local viscous_timescale / sample_ratio
cycles       = 4    # number of viscous_timescales of the outermost (slowest) annulus to span
                    # i.e. all annuli will have time-series spanning cycles*viscous_timescale(outer annulus)
sinusidal    = False
red_noise    = not(sinusidal)
Q            = 2.0  # width of lorentzian = 2 * F_visc / Q
incidence    = 0    # angle between line of sight and normal to disk plane
D            = 100 * pc # Distance of observer from the disk
checks       = True # Check inputs to functions eg R_in >= r >= R_out
prints       = True # print variables
annuli_ratio = (R_out / R_in) ** (1.0 / (number_of_annuli)) # determine the geometric ratio
observation_time = 1000


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
  - annulus is always a particular ring in the array of annuli 
  - N_dt is always an array of the number of time divisions per annulus  
    (how many divisions the annulus is split into)
  - dt is always an array of the time difference between these intervals
       (delta t)
  - start is an array of the start location for each annulus
  - i and j are used as counters in loops
  - where posible variables in procs are suffixed local_ 
            (except r,t and t_pos for ease of reading)
"""
'>>--------------------------------Functions--------------------------------<<'


def accretion_disk (number_of_annuli, R_in, R_out):
    """
    Creates an array of all the annuli (defined by calling accretion disk 
    array so annuli = R)
    Function "accretion_disk" to split the disk into number_of_annuli annuli   
    Annuli are geometrically distributed so that r/(r+1) is a constant 

    Inputs: Number of annuli, inner radius and outer radius of the accretion disk.
    Output: Array R of the inner radii of the annuli in the disk
    """
    
    if checks:     # sanity check of input to function
        if R_in >= R_out or number_of_annuli < 10:
            print ('ERROR in input to function:accretion_disk')
            
    # create an empty array:     
    local_annuli = np.empty(number_of_annuli)
    
    # loop through the annuli setting the radius for each
    for i in range(number_of_annuli):
        #R[i] = (((R_out - R_in) / number_of_annuli) * i) + R_in   # linear progression
        
        # annuli[i] is the inner radius of the i th annulus
        
        local_annuli[i] = R_in * (annuli_ratio ** i)                  # geometric progression
        
    return local_annuli


def viscous_timescale(r):    
    """
    Function "viscous_timescale" to calculate the viscous timescale at a particular radius 
    Using the formula given in Arevelo and Uttely
    alpha is the viscosity parameter which depends on the type of disk being measured
    
    
    Inputs: radius (in units of Rg)
    Output: viscous timescale at that radius (in units of Rt)
    """
    
    local_t_viscous = (2 * np.pi) * r ** (3.0 / 2.0) / ((H_over_R ** 2.0) * alpha)
    
    return local_t_viscous  


def phase_shift(local_annuli,local_annulus):
    """
    Function "Phase_Shift" to calculate the delta viscous velecity at a given
    annulus from the next outer one.
    Inputs: annulus and the annuli array
    Output: Delta Timescale
    """
    
    if checks:     # sanity check of input to function
        if local_annulus > (number_of_annuli - 1) or local_annulus < 0:
            print ('ERROR in input to function:phase_shift. Annulus must be within the array',local_annulus)
    
    # delta_t is the difference in time taken to get from an annuli to the centre
    # for two consecutive annuli and is therefore the time taken to get from the 
    # outer annuli to next annuli (delta_t is local to this function) 
        
    local_delta_t_visc = viscous_timescale(local_annuli[local_annulus+1]
    ) - viscous_timescale(local_annuli[local_annulus])
    return local_delta_t_visc



def time_division(local_annuli):
    """
    This calculates the number of time divisions required for each annulus
    sample ratio * viscous timescale for that annulus
    input: array of annuli radii
    outputs: three arrays (same length as the annuli array);
    
        local_N_dt_annulus = the number of time steps required to cover cycles * 
                viscous_timescale(outer annulus)

        local_dt_annulus = the corresponding time step for the local annulus
    
        local_annulus_t_pos_start = the start location of each annulus with a 
                variable array (sum of time divisions inside that annulus)
    
    
    """
    # radius of the outermost annulus in Rg
    local_r_max = local_annuli[number_of_annuli - 1]
    
    
    local_simulation_time = cycles * viscous_timescale(local_r_max)
    
    # this is the total simulation time;  cycles * the viscous timescale of the
    # outermost (hence, slowest) annulus. Therefore same for every annulus. 
    
    # Create empty arrays (number of divisions and start locations are integer values)
    # time divisions - number of time steps in each annulus. 
    
    local_N_dt_annulus = np.empty(number_of_annuli) 
    
    # annulus start position -  used to locate each annulus in arrays of mdot/Mdot
    
    local_annulus_t_pos_start = np.empty(number_of_annuli,dtype=int)
    
    # time intervals - difference in time between each time step different for 
    # each annulus. Measured in Rt. 
    
    local_dt_annulus = np.empty(number_of_annuli)
    
    #initialize both time series arrays and keep a count of the start of each annuli
    
    if prints:
        print ("time divisions")  
        
    count = 0
    for local_annulus in range(number_of_annuli):
        
        r = local_annuli[local_annulus]       #this is the radius of a given annulus in Rg
        
        local_annulus_t_pos_start[local_annulus] = count
        
        # time interval set to allow more entries where the timescale is longer. 
        
        local_N_dt_annulus[local_annulus] = local_simulation_time * sample_ratio / viscous_timescale(r)
        
        local_dt_annulus[local_annulus] = local_simulation_time / local_N_dt_annulus[local_annulus] 
        
        # update the running total to keep track of where the next annulus starts. 
        
        count = count + local_N_dt_annulus[local_annulus]
        
        if prints:
            print (local_annulus, count, local_N_dt_annulus[local_annulus],
                   local_dt_annulus[local_annulus], local_N_dt_annulus[local_annulus] 
                   * local_dt_annulus[local_annulus], local_simulation_time)
            
        
    return local_N_dt_annulus, local_dt_annulus, local_annulus_t_pos_start


def create_variable_array(local_N_dt_annulus):
    """
    This creates a 1D array with the time readings for annulus 0 followed by 1 
    and then 2 and so on. The number of entries is the sum of the time divisions
    per annulus. 
    Input: array of divisions
    Output: an empty array to hold all of these divisions
    """
    
    local_variable_array = np.empty(int(np.sum(local_N_dt_annulus)))
    
    return local_variable_array


def radius_to_annulus(r, local_annuli):    
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
                       
    local_annulus_smaller = int(np.log10(r / R_in) / np.log10(annuli_ratio))
    
        
    if local_annulus_smaller >= number_of_annuli :     # sanity check of input to function
        if checks:
            print ('Warning in function:radius to annulus. radius larger than R_out - set to last annulus')
        local_annulus_smaller = number_of_annuli-1

    return local_annulus_smaller

    
def read_variable_array(array,local_annuli,local_N_dt_annulus,local_dt_annulus,
                        local_annulus_t_pos_start,t,t_pos,r):
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
    local_annulus = radius_to_annulus(r,local_annuli)    
    local_annulus_start = local_annulus_t_pos_start[local_annulus]
    
    #check to determine is an exact position has been specified, if not reset t_pos based on t
    if t != -1:
        # t has been specified so calculate t_pos from it. 
        t_pos = int(t / local_dt_annulus[local_annulus])
        
    local_pos = local_annulus_start + t_pos
    
    if checks:     # sanity check of input to function
        if r > np.max(local_annuli) or r < np.min(local_annuli):
            print ('ERROR in input to function:read_variable_array. r out of range',r,local_annulus)
        if t_pos > local_N_dt_annulus[local_annulus]:
            print ('ERROR in input to function:read_variable_array. t greater than allowed in annuli',t,t_pos,local_annulus)
        if local_pos >= len(array):
            print ('ERROR in function:read_variable_array. t_pos out of range', local_annulus, local_annulus_start,
                      t_pos, local_N_dt_annulus[local_annulus] ) 
    
    #has an exact position has been specified? if not interpolate value between value1 and value2  
      
    if t != -1: 
        #interpolated result:
            
        value1 = array[local_pos]               # value before
        
        # check time position still in current annulus
        if t_pos + 1 < int(local_N_dt_annulus[local_annulus]):                     
            value2 = array[local_pos + 1]       # value after
        else:
            value2 = array[int(local_pos + 1 - local_N_dt_annulus[local_annulus])]                    # value after if looping around 
        # interpolate value from equation for linear interpolation. 
        value = value1 + ((value2 - value1) * (t  - (t_pos * local_dt_annulus[local_annulus]))
        )/ local_dt_annulus[local_annulus]

        if prints and value == 0:
            print ("read_zero_inter", local_annulus, value1, value2, t_pos, t, local_dt_annulus[local_annulus] )
        
    else:
        # if t_pos is known exactly then use the value. 

        value = array[local_pos]
        
        #if prints and value == 0:
        #    print ("read_zero_exact", local_annulus, t_pos, local_N_dt_annulus[local_annulus] )
    

        
    return value        


def update_variable_array(array,local_annuli,local_N_dt_annulus,local_dt_annulus,
                          local_annulus_t_pos_start,t,t_pos,r,value):    
    """
    generalised function to deal with updating the 1D array given radius and time inputs
    if t is given then it is used to to update the value before that time
    if t is set to -1 the t_pos position is used to update the value at that position
    Inputs: variable array to read, annuli, divisions, time intervals, start locations,
    time or time position (time takes precedence), radius and value to be placed in array
    Output: none 
    """
    
    #determine the annulus from the radius (needed to get the start of annulus position)
    local_annulus = radius_to_annulus(r,local_annuli)
    
    #check to determine is an exact position has been specified, if not reset t_pos based on t
    if t != -1:
        # t has been specified so calculate t_pos from it. 
        t_pos = int(t / local_dt_annulus[local_annulus])

    if checks:     # sanity check of input to function
        if r > np.max(local_annuli) or r < np.min(local_annuli):
            print ('ERROR in input to function:update_variable_array. r out of range', 
                   r, local_annulus)    
        if t_pos > local_N_dt_annulus[local_annulus]:
            print ('ERROR in input to function:update_variable_array. t greater than allowed in annuli',
                   t,t_pos,local_annulus)
        
    
    # Determine the start position of the annulus        
    local_annulus_start = local_annulus_t_pos_start[local_annulus]

    # set the value
    array[local_annulus_start + t_pos] = value
    return ()
    

def m_dot(local_annuli, local_N_dt_annulus, local_dt_annulus, local_annulus_t_pos_start):
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
    local_m_dot = create_variable_array(local_N_dt_annulus)
    
    # Choice of variability in m_dot sinusidal or red noise (power law/lorentzian)
    if sinusidal:
        for local_annulus in range(number_of_annuli):
            # For each annulus calculate the radius and viscous timescale
            # then for each time division use the timescale as the frequency of a sine wave
            r = local_annuli[local_annulus]
            timescale = viscous_timescale(r) / (2.0 * np.pi) 
            for t_pos in range(local_N_dt_annulus[local_annulus]):
                t = t_pos * local_dt_annulus[local_annulus]
                value = scale_factor * np.sin( t / timescale)
                update_variable_array(local_m_dot,local_annuli,local_N_dt_annulus
                                      ,local_dt_annulus,local_annulus_t_pos_start,-1,t_pos,r,value)
    if red_noise:
        for local_annulus in range(number_of_annuli):
            # For each annulus calculate the radius and use the lorentzian 
            # function to return the values for time divisions within that annulus. 
            r = local_annuli[local_annulus]
            if prints and ((local_annulus / 5) == int(local_annulus / 5)):
                print ("m_dot annulus ", local_annulus + 1 , " of ", number_of_annuli)
            # use function to return array of values 
            for i in range(cycles):
                local_startpos = local_annulus_t_pos_start[local_annulus] + i*int(
                        local_N_dt_annulus[local_annulus] /cycles)
                local_endpos = local_startpos + int(local_N_dt_annulus[local_annulus] /cycles)
                local_m_dot[local_startpos:local_endpos] = red_noise_factor * generate_lorenztian_law(
                        int(local_N_dt_annulus[local_annulus] / cycles), 
                        local_dt_annulus[local_annulus], Q, (1.0 / viscous_timescale(r)))
                
    return local_m_dot


def M_dot(local_m_dot,local_annuli,local_N_dt_annulus,local_dt_annulus,local_annulus_t_pos_start):
    """
    Function: "M_dot" to calculate the overall rate of change of mass in a particular 
    annulus. For inner annului the rate of flow is determined by the rate from the next
    outer annuli and the small change m_dot. the rate from the outer annuli is time delayed by
    the time it takes the mass to cross that annuli  (t_offset).The code cycles through
    the annuli from the outside (penultimate annulus) to the inside. As the pattern is 
    cyclical I assume that if the time is smaller than the offset we can start the cycle 
    again.
    
    Inputs: m_dot (small changes in annulus), annuli and local_N_dt_annulus,
            array of start locations for each annulus in the variable array
    Output: Array of Capital M_dot changes
    """
    #create the empty array
    local_M_dot = create_variable_array(local_N_dt_annulus)
    local_M_dot.fill(M0_dot)
    
    #For outer annulus there is no other input than the base rate of flow into
    #the disk (M0_dot) and the small change (m_dot) in that annuli 
    
    # Determine the radius of the outer annulus
    r_max = local_annuli[number_of_annuli-1]
    
    #if prints:
        #print ("r_max, t_pos, local_t_pos_m_dot, value")
        
    # For each time division in the array calculate M_dot and update the variable array
    for t_pos in range(int(local_N_dt_annulus[number_of_annuli-1])):
        
        
        local_t_pos_m_dot = read_variable_array(local_m_dot,local_annuli,local_N_dt_annulus,
                                                    local_dt_annulus,local_annulus_t_pos_start
                                                    ,-1,t_pos,r_max)
        
        value = M0_dot * (1.0 + local_t_pos_m_dot)
        
        #if prints:
            #print (int(r_max), t_pos, local_t_pos_m_dot, value)
            
        update_variable_array(local_M_dot,local_annuli,local_N_dt_annulus,
                              local_dt_annulus,local_annulus_t_pos_start,-1,t_pos,r_max,value)
        
    # For inner annuli the base rate of flow into
    # the annulus is the M_dot from the next outer annulus (offset by the 
    # time it takes to cross the annulus - phase_shift)
   
    if prints:
        print ("inner Mdot values")
        print ("r local_annulus t_offset_annulus local_dt_annulus local_N_dt_annulus")
        
    for local_annulus in range(number_of_annuli-2,-1,-1): # work backwards through array
    
        r  = local_annuli[local_annulus]      # this annulus
        r1 = local_annuli[local_annulus + 1]  # next outer annulus (from whence M_dot arrives)
        
        # Determine the time offset from the outer annulus
        
        t_offset_annulus = phase_shift(local_annuli,local_annulus)
        
        if prints:
            print (int(r), local_annulus, int(t_offset_annulus), int(local_dt_annulus[local_annulus]),
                   int(local_N_dt_annulus[local_annulus]))
            
        # For each time division calculate M_dot and update the variable array
        
        for t_pos in range(int(local_N_dt_annulus[local_annulus])):
                           
            # Change time position to proper time (Rt)
            
            t_offset = (t_pos * local_dt_annulus[local_annulus]) - t_offset_annulus 
            if t_offset < 0:    # If subtracting the offset makes the time negative loop round the timescale
                t_offset = t_offset + (local_N_dt_annulus[local_annulus]
                * local_dt_annulus[local_annulus])
                
            #while t_offset > local_N_dt_annulus[local_annulus] * local_dt_annulus[local_annulus]:
                #t_offset = t_offset - local_N_dt_annulus[local_annulus] * local_dt_annulus[local_annulus]
            

            local_t_pos_m_dot = read_variable_array(local_m_dot,local_annuli,local_N_dt_annulus,
                                                    local_dt_annulus,local_annulus_t_pos_start
                                                    ,-1,t_pos,r)
            
            local_next_t_pos_M_dot = read_variable_array(local_M_dot,local_annuli,local_N_dt_annulus,
                                                         local_dt_annulus,local_annulus_t_pos_start,
                                                         t_offset,-1,r1)
            
            value = local_next_t_pos_M_dot * (1.0 + local_t_pos_m_dot)
            
            if prints and value == 0:
                print ("zero", local_annulus, t_offset, t_pos, r1, radius_to_annulus(r1,local_annuli))
                
            update_variable_array(local_M_dot,local_annuli,local_N_dt_annulus,local_dt_annulus,
                                  local_annulus_t_pos_start,-1,t_pos,r,value)
            
    return local_M_dot


def effective_temperature(r,t_pos,t,local_M_dot,local_annuli,local_N_dt_annulus,
                          local_dt_annulus,local_annulus_t_pos_start): 
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
        local_M_dot_value = read_variable_array(local_M_dot,local_annuli,
                                                local_N_dt_annulus,local_dt_annulus,
                                                local_annulus_t_pos_start,-1,t_pos,r)
    else:
        #t_pos = -1
        local_M_dot_value = read_variable_array(local_M_dot,local_annuli,
                                                local_N_dt_annulus,local_dt_annulus,
                                                local_annulus_t_pos_start,t,-1,r)
        
    # Determine M_dot at the specified radius and time. Use this to calculate temperature
    #M_dot_value = read_variable_array(M_dot,annuli,local_N_dt_annulus,interval,start,-1,t_pos,r)
    local_T_eff = (3 * G * M_body * local_M_dot_value * (1 - ((R_in / r) ** 0.5)) / (
            (r ** 3) * 8 * np.pi * sigma)) ** 0.25
            
    # return the effective temperature adjusted for the units of Rg        
    return local_T_eff / (Rg ** 0.75)


def annulus_effective_temperature(local_M_dot,local_annuli,local_N_dt_annulus,
                                  local_dt_annulus,local_annulus_t_pos_start):
    """
    Effective Temperature, integrated across time for an annulus
    used in determining the blackbody spectrum. repeated for each annulus to 
    give the temperature across the disk
    Inputs: M_dot (total mass changes in annulus), annuli, 
    time divisions,time intervals and array of start locations for each annulus in the variable array
    Output:  array of effective temperatures (in Kelvin) per annulus
    """
    
    # Create an empty array to hold the values
    local_annuli_temperature = np.empty(number_of_annuli)
    
    # loop through the annuli to get a result for each annulus and add it to the array
    for local_annulus in range(number_of_annuli):
        
        # find the radius of the annulus and reset the counter to zero for each annulus
        r = local_annuli[local_annulus]
        local_total_effective_temp = 0
        
        # Loop through the time divisions calulating each temperature and adding it to the counter (total effective temp)
        for t_pos in range(int(local_N_dt_annulus[local_annulus])):
            local_total_effective_temp = local_total_effective_temp + effective_temperature(
                    r,t_pos,-1,local_M_dot,local_annuli,local_N_dt_annulus,
                    local_dt_annulus,local_annulus_t_pos_start)
            
        # Divide the total temperature by the number of readings to get the average    
        local_average_effective_temp = local_total_effective_temp / int(local_N_dt_annulus[local_annulus])
        
        # Update the array
        local_annuli_temperature[local_annulus] = local_average_effective_temp
        
    # Return the finished array of temperatures by annulus    
    return local_annuli_temperature

    
def flux_at_frequency(local_annuli,local_frequency,local_M_dot,local_N_dt_annulus,
                      local_dt_annulus,local_temperature):
    """
    Determine the flux for a specified frequency by integrating the BB radiation
    across the disk
    Inputs: annuli, the frequecncy to be used for the calculation, M_dot (total mass changes in annulus),  
    time divisions,time intervals and array of temperatures for each annulus 
    Output:  Flux from the disk at the specified frequency
    """
    
    # set the total to zero before adding components from each annulus
    local_total_integral = 0
    
    
    # Loop thgrough thye annuli adding the flux from each to the total
    for local_annulus in range(number_of_annuli):
        
        # integrating using dR (annulus thinkness)
        # dR is the difference between radii of adjacent annuli unless it is the
        # outer annulus which doesnt have an adjacent one, in this case we use 
        # the total disk radius as the outer radius. 
        if local_annulus != number_of_annuli - 1:
            dR = local_annuli[local_annulus+1] - local_annuli[local_annulus]
        else:
            dR = R_out - local_annuli[local_annulus]
            
        # For each annulus determine the radius and effective temperature
        r = local_annuli[local_annulus]
        local_T_eff = local_temperature[local_annulus]
        
        # Find the area under the curve for this annulus and add it to the total
        area = dR * r / (np.exp(h * local_frequency / (k * local_T_eff)) - 1)
        local_total_integral = local_total_integral + area
        
    # adjust the flus with the required constants from the equation and return it
    # units = watts/m^2
    
    local_flux = 4 * np.pi * np.cos(incidence) * (local_frequency ** 3
                                   ) * local_total_integral * h / ((c * D) ** 2)
    return local_flux     
       
  
def instantaneous_flux_at_frequency(local_annuli,local_frequency,local_M_dot,local_N_dt_annulus,
                                    local_dt_annulus,local_annulus_t_pos_start,t):
    """
    Determine the flux for a specified frequency by integrating the BB radiation
    across the disk
    Inputs: annuli, the frequecncy to be used for the calculation, M_dot (total mass changes in annulus),  
    time divisions,time intervals and array of temperatures for each annulus 
    Output:  Flux from the disk at the specified frequency
    """
    
    # set the total to zero before adding components from each annulus
    local_total_integral = 0
    
    
    # Loop thgrough thye annuli adding the flux from each to the total
    for local_annulus in range(number_of_annuli):
        
        # integrating using dR (annulus thinkness)
        # dR is the difference between radii of adjacent annuli unless it is the
        # outer annulus which doesnt have an adjacent one, in this case we use 
        # the total disk radius as the outer radius. 
        if local_annulus != number_of_annuli - 1:
            dR = local_annuli[local_annulus+1] - local_annuli[local_annulus]
        else:
            dR = R_out - local_annuli[local_annulus]
            
        # For each annulus determine the radius and effective temperature
        r = local_annuli[local_annulus]
        local_T_eff = effective_temperature(r,-1,t,local_M_dot,local_annuli,local_N_dt_annulus,
                                      local_dt_annulus,local_annulus_t_pos_start)
        
        # Find the area under the curve for this annulus and add it to the total
        local_area = dR * r / (np.exp(h * local_frequency / (k * local_T_eff)) - 1.0)
        local_total_integral = local_total_integral + local_area
        
    # adjust the flus with the required constants from the equation and return it
    # units = watts/m^2
    
    local_flux = 4 * np.pi * np.cos(incidence) * (local_frequency ** 3.0
                                   ) * local_total_integral * h / ((c * D) ** 2.0)
    return local_flux 


def generate_lorenztian_law(local_N, dt, beta, f_visc, generate_complex=False, random_state=None):
    
    """Generate a lorentzian-law light curve

    This uses the method from Timmer & Koenig [1]_

    Parameters
    ----------
    number_of_annuli : integer
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
    
    Npos = int(local_N / 2)
    domega = (2 * np.pi / (dt * local_N))

    #create lorenztian distribution (same as cauchy)
    # parameters are centre and width. 
    dist = cauchy(f_visc, f_visc / beta)
     
    if generate_complex:
        omega = domega * np.fft.ifftshift(np.arange(local_N) - int(local_N / 2))
    else:
        omega = domega * np.arange(Npos + 1)

    x_fft = np.zeros(len(omega), dtype=complex)
    
    #create gaussian distribution for real and imaginary parts
    x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
    x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

#    x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
#    x_fft[1:] *= (1. / np.sqrt(2))

# white noise (gaussian) multiplied by filter function (lorentzian)
    x_fft.real = dist.pdf(omega) * x_fft.real
    x_fft.imag = dist.pdf(omega) * x_fft.imag
#       x_fft[i] = lambda_visc2 / (x_fft[i] - angular_visc2 + lambda_visc2)

    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (local_N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, local_N)

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


def plot_mass_flow(local_annuli,local_N_dt_annulus,local_dt_annulus,
                   local_Mm_dot,local_annulus_t_pos_start,local_y_prefix):

    """
    Plot the mass flow (either m_dot or M_dot) for set annuli (start, middle and end of disk)
    Inputs: annuli,time divisions, time interval , M_dot or m_dot, annuli start positions and the prefix  
    to use on the plot and PDF.
    Output:  nothing returned, plot and PDF produced
    """
    
    for local_annulus in range(0,number_of_annuli,50):
#    for local_annulus in [1,int(number_of_annuli/2),8,number_of_annuli-1]:
    
    # create array of m or M dot for a single annulus by taking the start of the annulus
    # until the start of the next annulus and then back one. 
        if local_annulus < number_of_annuli - 1:
            local_annulus_Mm_dot = local_Mm_dot[local_annulus_t_pos_start[local_annulus]:local_annulus_t_pos_start[local_annulus+1]-1]
        else:
            local_annulus_Mm_dot = local_Mm_dot[local_annulus_t_pos_start[local_annulus]:]
        # actual time (Rt)
        time_axis = range(1, local_N_dt_annulus[local_annulus]) * local_dt_annulus[local_annulus]
        plt.plot(time_axis, local_annulus_Mm_dot, label=local_annulus)
        plt.xlabel("Time", fontsize =10)
        plt.xticks([])
        plt.yticks([])
        local_y_label = str(local_y_prefix) + " at Annulus " + str(local_annulus)
        file = local_y_label + ".pdf"
        plt.ylabel(local_y_label, fontsize =10)
        plt.savefig (file)
        plt.show()
    return ()
 
       
def plot_spectrum(local_annuli,local_N_dt_annulus,local_dt_annulus,
                   local_M_dot,local_annulus_t_pos_start,number_of_points, local_disk_temperature):

    """
    Plot the spectrum produced by the disk
    Inputs: annuli,time divisions, time interval , M_dot, annuli start positions
    number of points to plot and the array of annuli temperatures.
    Output:  nothing returned, plot and PDF produced
    """
    
    local_y_plot_values = np.empty(number_of_points) # set up empty array of y coordinates
    local_x_plot_values = np.empty(number_of_points) # set up empty array of y coordinates


# for each point calculate a frequency and use that frequency to calculate 
# the flux given by the disk. 
# loop through num records using i as the counter. 

    for i in range(number_of_points):
        # calculate frequency (expecting x rays)
        local_frequency = 2 * i * 1e16  
        local_x_plot_values[i] = (local_frequency)
        local_y_plot_values[i] = flux_at_frequency(local_annuli, local_frequency, local_M_dot, 
    local_N_dt_annulus, local_dt_annulus, local_disk_temperature)
        
    plt.xlabel("frequency (Hz)", fontsize =10)
    plt.ylabel("flux (Wm^-2) ", fontsize =10)
    plt.plot(local_x_plot_values, local_y_plot_values, 'r', linewidth=1, label ='$Cˆ{12}$' )
    plt.yticks([])  # turn off axis numbers on y axisto make graph clearer
    plt.savefig ('spectum.pdf')
    plt.show()
    return ()


def print_timestamp(local_message):
    
    """
    Print time stamp and message so we know what is running and how long it takes.
    Inputs: message
    Output:  nothing returned, timestamp and text message printed
    """
     
    now = datetime.datetime.now()
    print (now.strftime("%Y-%m-%d %H:%M:%S"), local_message)
    return ()
    
    
def lightcurve(local_annuli,local_N_dt_annulus,local_dt_annulus,
                   local_M_dot,local_annulus_t_pos_start,local_observation_time, local_frequency):
    
    """
    Creat lightcurve data
    Inputs: annuli,time divisions, time interval , M_dot, annuli start positions
    time each observation takes and the frequency being observed.
    Output:  arrays of counts and the time axis and plot / pdf
    """
    
    local_inner_time = int(local_dt_annulus[0]-1)  # maximum amount of divisions
    local_time_axis = np.empty(int(local_inner_time / local_observation_time)) # set up x value array
    local_counts    = np.empty(int(local_inner_time / local_observation_time)) # set up y value array

    for i in range(int(local_inner_time/local_observation_time)):  # loop through number of measurements
        local_observation_count = 0            # initialise observation count for each measurement
        for j in range(local_observation_time):     # loop through the time over which counts are taken 
            t = ((i * local_observation_time) + j) * local_dt_annulus[0] # convert tpos to time value
            local_observation_count = local_observation_count + instantaneous_flux_at_frequency(
                    local_annuli, local_frequency, local_M_dot,local_N_dt_annulus,
                    local_dt_annulus,local_annulus_t_pos_start,t)
            # observation count is accumulated by adding the flux at each time value. 
        local_counts[i] = local_observation_count / (h * local_frequency) # convert from energy to number of photons
        local_time_axis[i] = t                 # set time of measurement to be end of measurement.
    
    # plot time against counts which gives a basic lightcurve   
    plt.xlabel("time", fontsize =10)
    plt.ylabel("counts ", fontsize =10)
    plt.plot(local_time_axis, local_counts, 'r', linewidth=1, label ='$Cˆ{12}$' )
    plt.savefig ('lightcurve.pdf')
    plt.show()
    
    return local_counts, local_time_axis


'>>--------------------------------Main Code--------------------------------<<'

# print parameters and time stamp so we know what is running and how long it takes. 

print_timestamp("Annuli Number=" + str(number_of_annuli) + " cycles=" + str(cycles) + " samples=" + str(sample_ratio))

#these are the radii corresponding to the disk annuli in units of Rg

disk_annuli = accretion_disk(number_of_annuli, R_in, R_out)
   
# set up the time divisions and intervals and also the start locations of annuli 
# for the variable arrays. 

N_dt_annulus, dt_annulus, annulus_t_pos_start = time_division(disk_annuli)

print_timestamp("Status: disk (radius ,dt's, number of dt's and start locations) set up")

# plotting the radius of the disk against time divisions 
# this is only a test to check code plot isnt required. 

plt.plot(disk_annuli, N_dt_annulus)
plt.xlabel("Disk Radius", fontsize =10)
plt.ylabel("Time Divisions", fontsize =10)
plt.savefig ('time_divisons.pdf')
plt.show()

# set up variable array of m dot for every time position for every annulus
# (small variable mass movement wihin an annulus.)

m_dot = m_dot(disk_annuli,N_dt_annulus,dt_annulus,annulus_t_pos_start) 

print_timestamp("Status: m_dot set up")

# plot the m dot 

plot_mass_flow(disk_annuli,N_dt_annulus,dt_annulus,
                   m_dot,annulus_t_pos_start,"m_dot")    
    

# set up variable array of M dot for every time position for every annulus
# (overall mass movement wihin an annulus.)

M_dot = M_dot(m_dot,disk_annuli,N_dt_annulus,dt_annulus,annulus_t_pos_start) 

del m_dot
gc.collect()

print_timestamp("Status: M_dot set up")

# plotting the M dot 
 
plot_mass_flow(disk_annuli,N_dt_annulus,dt_annulus,
                   M_dot,annulus_t_pos_start,"M_dot")  

# set up array of effective temperature (K) for every annulus

disk_temperature = annulus_effective_temperature(M_dot,disk_annuli,N_dt_annulus,dt_annulus,annulus_t_pos_start) 


print_timestamp("Status: Temperature set up")

 # plot radius against temperature (not needed used as a check)
 
plt.plot(disk_annuli, disk_temperature)
plt.xlabel("Disk Radius", fontsize =10)
plt.ylabel("Temperature", fontsize =10)
plt.savefig ('temperature.pdf')
plt.show()


# using all the data modelled on the disk above (temperature, radius, Mdot, mdot etc)
# we can now start using the model to make prediction eg flux from the disk. 

plot_spectrum(disk_annuli,N_dt_annulus,dt_annulus, M_dot,annulus_t_pos_start, 1000, disk_temperature)

print_timestamp("Status:Spectrum produced")

# create and plot lightcurve data (counts in each observation period against time)

counts, time_axis = lightcurve(disk_annuli,N_dt_annulus,dt_annulus,
                   M_dot,annulus_t_pos_start, observation_time, 0.6e20)
#frequency       = 0.6e20 # x-ray
    
print_timestamp("Status: Lightcurve Produced")

# use mathplot.psd to give a power spectrum density and return pds and frequency arrays
pds, freqs =  plt.psd(counts, 512, 1 / observation_time) 
plt.savefig ('plot.psd.pdf')
plt.show()

# replot using the log of the frequency
log_freq = np.log10(freqs)
plt.xlabel("Log Frequency", fontsize =10)
plt.ylabel("PSD ", fontsize =10)
plt.plot(log_freq,pds,'r',linewidth=1, label ='$Cˆ{12}$' )
plt.savefig ('plot.logpsd.pdf')
plt.show()

# replot also using log of pds
log_pds = np.log10(pds)
plt.xlabel("Log Frequency", fontsize =10)
plt.ylabel("Log PSD ", fontsize =10)
plt.plot(log_freq,log_pds,'r',linewidth=1, label ='$Cˆ{12}$' )
plt.savefig ('plot.logfreq.logpsd.pdf')
plt.show()

# use periodogram method and return frequency and power density:
    
f, Pxx_den = signal.periodogram(counts,observation_time)
plt.semilogy(np.log10(f), Pxx_den) #power density already log
#plt.ylim([1e-7, 1e2])
plt.xlabel('log frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.savefig ('logsemilogy.pdf')
plt.show()

print_timestamp("array size = " + str(len(M_dot)))
#