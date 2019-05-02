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
M_sol = 1.99e30         # 1 solar mass (kg)
years = 3600*24*365.25  # seconds in a year 
pc    = 3.086e16        # parsecs in metres

#----------------------------------------------------------------------------
'>>--------------------------------Parameters-------------------------------<<'

R_in         = 10   # Radius of Central Body, eg. Black Hole, also the start of the disk  (Rg)
M_body       = 1 * M_sol  # Mass of Central Body in solar masses (kg)
R_out        = 1000.0  # Radial distance to edge of disk (Rg)
number_of_annuli = 3 # Number of Annuli , N > 1 at all times 
print_plot_interval = int(number_of_annuli / 50)
M0_dot       = 1e-16 * M_sol / years # Accretion Rate at outer radii, where mass first flows in (kg/s) 
alpha        = 1.0      # Viscosity Parameter 
H_over_R     = 0.01     # ratio of H to R
scale_factor = 0.1      # m_dot << 1
#red_noise_factor = 1e-8 # factor to scale up red_noise
sample_ratio = 30       # this parameter sets the time resolution of the time-series for each annulus
                        # must be greater than 9
                        # for each annulus, our time step is the local viscous_timescale / sample_ratio
sinusidal    = False
red_noise    = not(sinusidal)
Q            = 2.0      # width of lorentzian = 2 * F_visc / Q
incidence    = 0        # angle between line of sight and normal to disk plane
D            = .001 * pc # Distance of observer from the disk
checks       = False     # Check inputs to functions eg R_in >= r >= R_out
prints       = False     # print variables
plot_all     = False    # produce check plots?
debug        = False    # some more debugging print outs...
debug2       = False    # some more debugging print outs...
annuli_ratio = (R_out / R_in) ** (1.0 / (number_of_annuli)) # determine the geometric ratio

'>>--------------------------------Units------------------------------------<<'

Rg   = (G * M_body) / (c * c)  # Unit of Distance (Arevalo & Uttley)
Rt   = Rg / c                  # Unit of Time     (Arevalo & Uttley)
#D    = D / Rg                 # put distance to observer into these units
#let's not do that -- the distance is only used for fluxes...

#----------------------------------------------------------------------------
'>>-------------------------------Conventions-------------------------------<<'
"""
  - r is always a single radius in Rg units
  - t is always a single time in Rt units
  - tpos is always the single time position in a variable array of Mdot/mdot - 
       can be converted to t by multiplying by the time interval in that annulus
  - Where t and tpos are inputs to a function t will be used unless it is set to -1
  - annuli is the array of the inner radii of the annuli the disk is split into.  
  - annulus is always a particular ring in the array of annuli 
  - N_dt is always an array of the number of time divisions per annulus  
    (how many divisions the annulus is split into)
  - dt is always an array of the time difference between these intervals
       (delta t)
  - start is an array of the start location for each annulus
  - i and j are used as counters in loops
  - where posible variables in procs are suffixed local_ 
            (except r,t and tpos for ease of reading)
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
    
    # delta_t_visc is the difference in time taken to get from an annuli to the centre
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
    
        local_annulus_tpos_start = the start location of each annulus with a 
                variable array (sum of time divisions inside that annulus)

        local_simulation_time = the total time covered by our simulation
    """
    
    # radius of the outermost annulus in Rg
    local_r_max = local_annuli[number_of_annuli - 1]
    
    local_simulation_time = viscous_timescale(local_r_max)
    
    # this is the total simulation time;  cycles * the viscous timescale of the
    # outermost (hence, slowest) annulus. Therefore same for every annulus. 
    
    # Create empty arrays (number of divisions and start locations are integer values)
    # time divisions - number of time steps in each annulus. 
    
    local_N_dt_annulus = np.empty(number_of_annuli) 
    
    # annulus start position -  used to locate each annulus in arrays of mdot/Mdot
    
    local_annulus_tpos_start = np.empty(number_of_annuli,dtype=int)
    
    # time intervals - difference in time between each time step different for 
    # each annulus. Measured in Rt. 
    
    local_dt_annulus = np.empty(number_of_annuli)
    
    #initialize both time series arrays and keep a count of the start of each annuli
    
    if prints:
        print ("time divisions")  
        
    count = 0
    for local_annulus in range(number_of_annuli):
        
        r = local_annuli[local_annulus]       #this is the radius of a given annulus in Rg
        
        local_annulus_tpos_start[local_annulus] = count
        
        # time interval set to allow more entries where the timescale is longer. 
        
        local_N_dt_annulus[local_annulus] = local_simulation_time * sample_ratio / viscous_timescale(r)
        
        local_dt_annulus[local_annulus] = local_simulation_time / local_N_dt_annulus[local_annulus] 
        
        # update the running total to keep track of where the next annulus starts. 
        
        count = count + local_N_dt_annulus[local_annulus]
        
        if prints and (local_annulus / print_plot_interval == 
                       int(local_annulus / print_plot_interval)):
            print (local_annulus, count, local_N_dt_annulus[local_annulus],
                   local_dt_annulus[local_annulus], local_N_dt_annulus[local_annulus] 
                   * local_dt_annulus[local_annulus], local_simulation_time)
            
        
    return local_N_dt_annulus, local_dt_annulus, local_annulus_tpos_start,local_simulation_time


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
                       
    local_annulus_smaller = int((np.log10(r / R_in) +.0001) / np.log10(annuli_ratio))
    
        
    if local_annulus_smaller >= number_of_annuli :     # sanity check of input to function
        if checks:
            print ('Warning in function:radius to annulus. radius larger than R_out - set to last annulus')
        local_annulus_smaller = number_of_annuli-1

    return local_annulus_smaller

    
def read_variable_array(array,local_annuli,local_N_dt_annulus,local_dt_annulus,
                        local_annulus_tpos_start,t,tpos,r):
    """
    generalised function to deal with treating the 1D array given specific 
    radius and time inputs. 
    if t is given then it is used to to give an linearly interpolated value between
    the two nearest timestamps.
    if t is set to -1 the tpos position is used to give the value at that position.
    Inputs: variable array to read, annuli, divisions, time intervals, start locations,
    time or time position (time takes precedence) and radius
    Output: value of the given variable array at the radius and time specified    
    """
    
    # determine the annulus from the radius and find its start location
    local_annulus = radius_to_annulus(r,local_annuli)    
    local_annulus_start = local_annulus_tpos_start[local_annulus]
    
    #check to determine is an exact position has been specified, if not reset tpos based on t
    if t != -1:
        # t has been specified so calculate tpos from it. 
        tpos = int(t / local_dt_annulus[local_annulus])
        tpos_decimal = t / local_dt_annulus[local_annulus]
        
    local_pos = local_annulus_start + tpos
    
    if checks:     # sanity check of input to function
        if r > (np.max(local_annuli) + 0.1) or r < np.min(local_annuli):
            print ('ERROR in input to function:read_variable_array. r out of range',r,local_annulus)
        if tpos > local_N_dt_annulus[local_annulus]:
            print ('ERROR in input to function:read_variable_array. t greater than allowed in annuli',t,tpos,local_annulus)
        if local_pos >= len(array):
            print ('ERROR in function:read_variable_array. tpos out of range', local_annulus, local_annulus_start,
                      tpos, local_N_dt_annulus[local_annulus] ) 
    
       
    #has an exact position has been specified? if not interpolate value between value1 and value2  
      
    if t != -1: 
        #interpolated result:
            
        value1 = array[local_pos]               # value before
        
        # check time position still in current annulus
        if tpos_decimal + 1 < local_N_dt_annulus[local_annulus]:                     
            value2 = array[local_pos + 1]       # value after
        else:
            value2 = array[local_pos + 1 - int(local_N_dt_annulus[local_annulus])]                    # value after if looping around 
        # interpolate value from equation for linear interpolation. 
        value = value1 + ((value2 - value1) * (t  - (tpos_decimal * local_dt_annulus[local_annulus]))
        / local_dt_annulus[local_annulus])

        if prints and value == 0:
            print ("read_zero_inter", local_annulus, value1, value2, tpos, t, local_dt_annulus[local_annulus] )
        
    else:
        # if tpos is known exactly then use the value. 

        value = array[local_pos]
        
        #if prints and value == 0:
        #    print ("read_zero_exact", local_annulus, tpos, local_N_dt_annulus[local_annulus] )
    

        
    return value        


def update_variable_array(array,local_annuli,local_N_dt_annulus,local_dt_annulus,
                          local_annulus_tpos_start,t,tpos,r,value):    
    """
    generalised function to deal with updating the 1D array given radius and time inputs
    if t is given then it is used to to update the value before that time
    if t is set to -1 the tpos position is used to update the value at that position
    Inputs: variable array to read, annuli, divisions, time intervals, start locations,
    time or time position (time takes precedence), radius and value to be placed in array
    Output: none 
    """
    
    #determine the annulus from the radius (needed to get the start of annulus position)
    local_annulus = radius_to_annulus(r,local_annuli)
    
    #check to determine is an exact position has been specified, if not reset tpos based on t
    if t != -1:
        # t has been specified so calculate tpos from it. 
        tpos = int(t / local_dt_annulus[local_annulus])

    if checks:     # sanity check of input to function
        if r > (np.max(local_annuli) + 0.1) or r < np.min(local_annuli):
            print ('ERROR in input to function:update_variable_array. r out of range', 
                   r, local_annulus)    
        if tpos > local_N_dt_annulus[local_annulus]:
            print ('ERROR in input to function:update_variable_array. t greater than allowed in annuli',
                   t,tpos,local_annulus)
        
    
    # Determine the start position of the annulus        
    local_annulus_start = local_annulus_tpos_start[local_annulus]

    # set the value
    array[local_annulus_start + tpos] = value
    return ()
    

def m_dot(local_annuli, local_N_dt_annulus, local_dt_annulus, local_annulus_tpos_start):
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
            for tpos in range(int(local_N_dt_annulus[local_annulus])):
                t = tpos * local_dt_annulus[local_annulus]
                value = scale_factor * np.sin( t / timescale)
                update_variable_array(local_m_dot,local_annuli,local_N_dt_annulus
                                      ,local_dt_annulus,local_annulus_tpos_start,-1,tpos,r,value)

    if red_noise:
        for local_annulus in range(number_of_annuli):
            
            # For each annulus calculate the radius and use the lorentzian 
            # function to return the values for time divisions within that annulus. 
            
            r = local_annuli[local_annulus]
            if prints and ((local_annulus / print_plot_interval) == 
                           int(local_annulus / print_plot_interval)):
                print ("m_dot annulus ", local_annulus + 1 , " of ", number_of_annuli)
                
            # use function to return array of values 
            
            
            local_startpos = local_annulus_tpos_start[local_annulus] 
            local_endpos = local_startpos + int(local_N_dt_annulus[local_annulus])
            local_m_dot[local_startpos:local_endpos] = generate_lorenztian_law(
                    int(local_N_dt_annulus[local_annulus] ), 
                    local_dt_annulus[local_annulus], Q, (1.0 / viscous_timescale(r)))
            
            # standardise red noise to be in the range 0.01 - 0.09 i.e. <<1 but still significant
            
            red_noise_range = np.max(local_m_dot[local_startpos:local_endpos]) - np.min(local_m_dot[local_startpos:local_endpos]) 
            red_noise_factor = int(np.log10(red_noise_range)) + 2
            local_m_dot[local_startpos:local_endpos] = local_m_dot[local_startpos:local_endpos] * 10**(-red_noise_factor)
    
    return local_m_dot


def M_dot(local_m_dot,local_annuli,local_N_dt_annulus,local_dt_annulus,local_annulus_tpos_start):
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
    #create the empty array and initialise to M0_dot (mass rate into the disk)
    
    local_M_dot = create_variable_array(local_N_dt_annulus)
    local_M_dot.fill(M0_dot)
    
    #For outer annulus there is no other input than the base rate of flow into
    #the disk (M0_dot) and the small change (m_dot) in that annuli 
    
    # Determine the radius of the outer annulus
    
    local_r_max = local_annuli[number_of_annuli - 1]
    
    #if prints:
        #print ("r_max, tpos, local_tpos_m_dot, value")
        
    # For each time division in the array calculate M_dot and update the variable array

    # determine the annulus from the radius and find its start location
    local_annulus = radius_to_annulus(local_r_max,local_annuli)    
    local_annulus_start = local_annulus_tpos_start[local_annulus]

    tpos_start = local_annulus_start
    tpos_end = tpos_start + int(local_N_dt_annulus[number_of_annuli-1]) 
    local_tpos_m_dot = local_m_dot[tpos_start:tpos_end]
    local_M_dot1 = M0_dot * (1.0 + local_tpos_m_dot)
    local_M_dot[tpos_start:tpos_end] = local_M_dot1
    print(local_M_dot,len(local_M_dot))


    print(local_M_dot,len(local_M_dot))

    # For inner annuli the base rate of flow into
    # the annulus is the M_dot from the next outer annulus (offset by the 
    # time it takes to cross the annulus - phase_shift)
    
    if prints:
        print ("Inner M dot values")
        print ("r local_annulus t_offset_annulus local_dt_annulus local_N_dt_annulus")
        
    for local_annulus in range(number_of_annuli-2,-1,-1): # work backwards through array
        local_annulus_next = local_annulus + 1    
        
        local_total_time = local_N_dt_annulus[local_annulus] * local_dt_annulus[local_annulus]

        r  = local_annuli[local_annulus]          # this annulus
        #r1 = local_annuli[local_annulus + 1]      # next outer annulus (from whence M_dot arrives)
        
        # find the start locations for this annulus and the next outer one
        local_annulus_start = local_annulus_tpos_start[local_annulus]
        local_annulus_next_start = local_annulus_tpos_start[local_annulus_next]

        # Determine the time offset between these two annuli
        t_offset_annulus = phase_shift(local_annuli,local_annulus)
        
        if prints and ((local_annulus / print_plot_interval) == 
                       int(local_annulus / print_plot_interval)):
            print (int(r),"      ", local_annulus, "      ", int(t_offset_annulus), "      " , 
                   int(local_dt_annulus[local_annulus]), "      ", int(local_N_dt_annulus[local_annulus]))
            
        # For each time division calculate M_dot and update the variable array

        #first, work out the starting and ending indices for the current annulus
        tpos_start = local_annulus_start
        tpos_end = tpos_start + int(local_N_dt_annulus[local_annulus])
        #
        #now create an array of all indices that are associated with the current annulus
        tpos = np.arange(tpos_start,tpos_end)
        #
        #now construct the array of local m_dots
        local_tpos_m_dot = local_m_dot[tpos_start:tpos_end]

        
        #since Mdot(R_i) depends on mdot(r_i) and Mdot(R_i + 1) [the next annulus further out]
        #we now need to get local_next_tpos_Mdot
        #the complication here here is that we need to account for the time shift between
        #annuli

        #first, work out the actual time associated with each of the indices for the current
        #annulus, and then account for the shift between annuli
        t_offset = ((tpos-tpos_start) * local_dt_annulus[local_annulus]) - t_offset_annulus
        #now test if the resulting time is positive; fix it if it is not
        alternative1 = t_offset + local_total_time
        t_offset = np.where(t_offset > 0, t_offset, alternative1)
        #now test if the resulting time is beyond the max time; fix it if it is
        alternative2 = t_offset - local_total_time
        t_offset = np.where(t_offset <  local_total_time, t_offset, alternative2)
        #print('t_offset 3 =',t_offset)


        #now we need to construct the indices in local_M_dot that bracket 
        #these offset times, in the segment of that array that corresponds to the
        #*next* annulus (the next one further out, i.e. really the "previous" one)
        
        #we start by working out the start and end positions of the next annulus further out.
        tpos_next_start = local_annulus_next_start
        tpos_next_end = tpos_next_start + int(local_N_dt_annulus[local_annulus_next])

        #now let's get the full grid of those indices
        tpos_next = np.arange(tpos_next_start,tpos_next_end)

        #we can then convert these indices to proper times
        t_next = local_dt_annulus[local_annulus_next] * (tpos_next - tpos_next_start)

        #OK, now t_offset_annulus is the amount by which we need to offset the two annuli.
        #this means we need to find the indices in t_next that bracket the shifted times of
        #the current annulus. then we want to interpolate on Mdot, using these indices
        #
        #an easy -- but not the most efficient! -- way to accomplish this is to use
        #np.searchsorted(a,b). This finds the indices in the sorted array a where you'd have to
        #insert the elements of b in order for things to stay sorted. so the result of
        #searchsorted is the upper bracketing indices.
        #
        index_hi = np.searchsorted(t_next,t_offset)
        index_hi = np.where(index_hi < len(t_next)-1, index_hi, len(t_next)-1)
        index_hi = np.where(index_hi > 0, index_hi, 1)
        index_lo = index_hi - 1

        
        t_next_lo = np.take(t_next,index_lo)
        t_next_hi = np.take(t_next,index_hi)
            
        #and the corresponding indices in the overall array
        tpos_next_lo = np.take(tpos_next,index_lo)
        tpos_next_hi = np.take(tpos_next,index_hi)
        
        #and also the corresponding Mdots
        local_next_M_dot_lo = np.take(local_M_dot,tpos_next_lo)
        local_next_M_dot_hi = np.take(local_M_dot,tpos_next_hi)

        #    #now we can work out the interpolation fraction
        frac = (t_offset-t_next_lo) / (t_next_hi - t_next_lo)
        #except:
            
        #and hence the interpolated Mdot value in the next annulus
        local_next_M_dot = local_next_M_dot_lo + frac*(local_next_M_dot_hi - local_next_M_dot_lo)
        

        #so, now, finally, we can use this to update the local Mdot
        local_M_dot[tpos_start:tpos_end] = local_next_M_dot * (1.0 + local_m_dot[tpos_start:tpos_end])
        
    return local_M_dot


def effective_temperature(r,tpos,t,local_M_dot,local_annuli,local_N_dt_annulus,
                          local_dt_annulus,local_annulus_tpos_start): 
    """
    Effective Temperature at a particular radius and time position 
    Inputs: radius, time, time position,M_dot (total mass changes in annulus), annuli, 
    time divisions,time intervals and array of start locations for each annulus in the variable array
    Output: Local effective temperature (in Kelvin)
    """
    
    # check to determine is an exact position has been specified, if not reset tpos based on t
    if tpos != -1:
        # t has been specified so calculate tpos from it. 
        # Determine M_dot at the specified radius and time. Use this to calculate temperature
        local_M_dot_value = read_variable_array(local_M_dot,local_annuli,
                                                local_N_dt_annulus,local_dt_annulus,
                                                local_annulus_tpos_start,-1,tpos,r)
    else:
        #tpos = -1
        local_M_dot_value = read_variable_array(local_M_dot,local_annuli,
                                                local_N_dt_annulus,local_dt_annulus,
                                                local_annulus_tpos_start,t,-1,r)
        
    # Determine M_dot at the specified radius and time. Use this to calculate temperature
    #M_dot_value = read_variable_array(M_dot,annuli,local_N_dt_annulus,interval,start,-1,tpos,r)
    local_T_eff = (3 * G * M_body * local_M_dot_value * (1 - ((R_in / r) ** 0.5)) / (
            (r ** 3) * 8 * np.pi * sigma)) ** 0.25
            
    # return the effective temperature adjusted for the units of Rg        
    return local_T_eff / (Rg ** 0.75)


def annulus_effective_temperature(local_M_dot,local_annuli,local_N_dt_annulus,
                                  local_dt_annulus,local_annulus_tpos_start):
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
        
        # Loop through the time divisions calulating each temperature and adding it to the counter (total effective temp)
                #first, work out the starting and ending indices for the current annulus
        tpos_start = local_annulus_tpos_start[local_annulus]
        tpos_end = tpos_start + int(local_N_dt_annulus[local_annulus])
        #
        #now construct the array of local M_dots and their associated temperature
        local_tpos_M_dot = local_M_dot[tpos_start:tpos_end]
        
        local_T_eff = (3 * G * M_body * local_tpos_M_dot * (1 - ((R_in / r) ** 0.5)) / (
            (r ** 3) * 8 * np.pi * sigma)) ** 0.25
        
            
        # get the average for this annuli    
        local_average_effective_temp = np.mean(local_T_eff)
        
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


def annulus_lightcurve_at_frequency(r, delta_r, tinput, Mdot, local_frequency, toutput):
    """create the light curve of a single annulus at a given frequency and on a given time grid
    INPUT: 
      r = the radius of the current annulus
      dr = the width of the current annulus
      tinput = native time array for this annulus
      Mdot = the Mdot "light curve" on the native time array for this annulus
      local_frequency = the desired frequency for which the light curve should be computed
      toutput = the desired time array for the output light curve
    OUTPUT:
      light = the output light curve
    """
    #first, we calculate an array of effective temperatures for this annulus, i.e. the seqeuence
    #of temperature corresponding to the sequence of Mdots.
    #
    #we need radii in PHYSICAL units
    rphys = r*Rg
    delta_rphys = delta_r*Rg
    
    teff = (3 * G * M_body * Mdot * (1 - ((R_in / r) ** 0.5)) / (
            (rphys ** 3) * 8 * np.pi * sigma)) ** 0.25

    
    #teff = teff/1.e3
    if (debug2):
        #print('annulus at radius = ',r)
        #print('teff grid ',teff)
        print('M_body = ',M_body)
        print('Mdot[0] = ',Mdot[0])
        print('Rin = ',R_in)
        print('G = ',G)
        print('sigma = ',sigma)
        print('teff[0] = ',teff[0])
        
    
    #CK: I've not checked the following code, which is pretty much taken straight from
    #    "instantaneous_flux_at_frequency"
    # Find the area under the curve for this annulus and add it to the total
    #
    local_flux = delta_rphys * rphys / (np.exp(h * local_frequency / (k * teff)) - 1.0)
    local_flux = local_flux * 4 * np.pi * np.cos(incidence) * (local_frequency ** 3.0) * h / ((c * D) ** 2.0)

    #local_flux should now be an array on the native time grid for this annulus, so now
    #we interpolate it onto the desired time grid
    #
    if (debug):
        print('Length of tinput: ',len(tinput))
        print('Length of local_flux: ',len(local_flux))
        print('Length of Mdot: ',len(Mdot))
    light = np.interp(toutput,tinput,local_flux)
    
    return light

  
def instantaneous_flux_at_frequency(local_annuli,local_frequency,local_M_dot,local_N_dt_annulus,
                                    local_dt_annulus,local_annulus_tpos_start,t):
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
                                      local_dt_annulus,local_annulus_tpos_start)
        
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
                   local_Mm_dot,local_annulus_tpos_start,local_y_prefix):

    """
    Plot the mass flow (either m_dot or M_dot) for set annuli (start, middle and end of disk)
    Inputs: annuli,time divisions, time interval , M_dot or m_dot, annuli start positions and the prefix  
    to use on the plot and PDF.
    Output:  nothing returned, plot and PDF produced
    """
    
    for local_annulus in range(0,number_of_annuli, print_plot_interval):
#    for local_annulus in [1,int(number_of_annuli/2),8,number_of_annuli-1]:
    
    # create array of m or M dot for a single annulus by taking the start of the annulus
    # until the start of the next annulus and then back one. 
        if local_annulus < number_of_annuli - 1:
            local_annulus_Mm_dot = local_Mm_dot[local_annulus_tpos_start[local_annulus]:local_annulus_tpos_start[local_annulus+1]-1]
        else:
            local_annulus_Mm_dot = local_Mm_dot[local_annulus_tpos_start[local_annulus]:]
        # actual time (Rt)
        #time_axis = range(0, int(local_N_dt_annulus[local_annulus])) * local_dt_annulus[local_annulus]
        plt.plot(local_annulus_Mm_dot, label=local_annulus)
        #plt.plot(time_axis, local_annulus_Mm_dot, label=local_annulus)
        plt.xlabel("Time", fontsize =10)
        plt.xticks([])
        #plt.yticks([])
        local_y_label = str(local_y_prefix) + " at Annulus " + str(local_annulus)
        file = local_y_label + ".pdf"
        plt.ylabel(local_y_label, fontsize =10)
        plt.savefig (file)
        plt.show()
    return ()
 
       
def plot_spectrum(local_annuli,local_N_dt_annulus,local_dt_annulus,
                   local_M_dot,local_annulus_tpos_start,number_of_points, local_disk_temperature):

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
                   local_M_dot,local_annulus_tpos_start,disk_temp, local_frequency):
    
    """
    Create the light curve generated by the disk a local_frequency.
    Here is the idea:
       * start with the first observation time we are interested in
       * we know from local_M_dot what Mdot is at this time for each annulus
       * since flux depends on Mdot, we also know the flux contribution of each annulus at this time.
       * so we can work out the total flux at this time by summer out the flux contributions of all annuli
       * and we can then repeat this for all of the times we're interested in.
    The biggest challenge is avoiding loops, since they are slow in python.

    We're going to create the ouput light curve at the time resolution of the innermost annulus (i.e.
    the *full* -- fastest -- time resolution of the simulation).

    Inputs: annuli,time divisions, time interval , M_dot, annuli start positions, the grid of time-averaged 
            effective temperatures for each annulus, and the frequency being observed.
    Output:  arrays of counts and observation times, also the plot / pdf of the light curve
    """

    #first, we set up the observation time array (which is the same as the time array for the innermost annulus)
    #
    #the innermost anulus has index zero
    #
    obs_end_time = local_N_dt_annulus[0]*local_dt_annulus[0]
    #this is the final entry in the time array of the innermost annulus, and hence also of our observation time array
    #
    local_observation_times = np.arange(0.0,obs_end_time,local_dt_annulus[0])
    #this is the grid of observation times
    #
    total_flux_light_curve = local_observation_times*0.0
    #initializing the overall light curve

    #now we try to avoid uncessary looping over the huge time array. What we do is instead loop over the *annuli*.
    #for each annulus, we'll create a light curve sampled on the observation time grid
    #we can then just add these up to create the total disk light curve
    #

    
    for current_annulus_index in range(len(local_annuli)):
        if (current_annulus_index%100)==0:
            print('creating light curve produced by annulus ',current_annulus_index)
            
        #let's grab the light curve for this annulus, sampled onto our desired observation time grid
        #
        #first let's set up our inputs
        local_r = local_annuli[current_annulus_index]
        
        if current_annulus_index != number_of_annuli - 1:
            local_deltar = local_annuli[current_annulus_index+1] - local_annuli[current_annulus_index]
        else:
            local_deltar = R_out - local_annuli[current_annulus_index]

        time_step_current_annulus = local_dt_annulus[current_annulus_index]
        end_time_current_annulus = local_N_dt_annulus[current_annulus_index] * time_step_current_annulus
        N_timesteps_current_annulus = int(local_N_dt_annulus[current_annulus_index])
        trunc_end_time = int(local_N_dt_annulus[current_annulus_index]) * time_step_current_annulus
        
        #time_array_current_annulus = np.arange(0.0,end_time_current_annulus,time_step_current_annulus)
        time_array_current_annulus = np.linspace(0.0,trunc_end_time,N_timesteps_current_annulus)
        start_index_in_Mdot_array = local_annulus_tpos_start[current_annulus_index]
        stop_index_in_Mdot_array = start_index_in_Mdot_array + int(local_N_dt_annulus[current_annulus_index])

        if (debug):
            print('Simulation Time = ',simulation_time)
            print('Current End Time = ',end_time_current_annulus)
            print('Last Entry in Current Time Array =',time_array_current_annulus[-1])
            print('N1 = ',N_timesteps_current_annulus)
            print('N2 = ',int(local_N_dt_annulus[current_annulus_index]))
            print('N2f = ',local_N_dt_annulus[current_annulus_index])
        Mdot_array_current_annulus = local_M_dot[start_index_in_Mdot_array:stop_index_in_Mdot_array]
            
        local_flux_light_curve = annulus_lightcurve_at_frequency(local_r, local_deltar, time_array_current_annulus,
                                         Mdot_array_current_annulus,
                                         local_frequency, local_observation_times)
        total_flux_light_curve = total_flux_light_curve + local_flux_light_curve
        
    # plot time against counts which gives a basic lightcurve   
    plt.xlabel("time", fontsize =10)
    plt.ylabel("counts ", fontsize =10)
    plt.plot(local_observation_times, total_flux_light_curve, 'r', linewidth=1, label ='$Cˆ{12}$' )
    plt.savefig ('lightcurve.pdf')
    plt.show()
    
    return total_flux_light_curve, local_observation_times


'>>--------------------------------Main Code--------------------------------<<'

# print parameters and time stamp so we know what is running and how long it takes. 

print_timestamp("Annuli Number=" + str(number_of_annuli) + " samples=" + str(sample_ratio)
+ " alpha=" + str(alpha) + " H/R=" + str(H_over_R)+ " Q=" + str(Q))

#these are the radii corresponding to the disk annuli in units of Rg

disk_annuli = accretion_disk(number_of_annuli, R_in, R_out)
   
# set up the time divisions and intervals and also the start locations of annuli 
# for the variable arrays. 

N_dt_annulus, dt_annulus, annulus_tpos_start, simulation_time = time_division(disk_annuli)

print_timestamp("Status: disk (radius ,dt's, number of dt's and start locations) set up")

# plotting the radius of the disk against time divisions 
# this is only a test to check code plot isnt required. 

if (plot_all):
    plt.plot(disk_annuli, N_dt_annulus)
    plt.xlabel("Disk Radius", fontsize =10)
    plt.ylabel("Time Divisions", fontsize =10)
    plt.savefig ('time_divisons.pdf')
    plt.show()

# set up variable array of m dot for every time position for every annulus
# (small variable mass movement wihin an annulus.)

m_dot = m_dot(disk_annuli,N_dt_annulus,dt_annulus,annulus_tpos_start) 

print_timestamp("Status: m_dot set up")

# plot the m dot 

if (plot_all):
    plot_mass_flow(disk_annuli,N_dt_annulus,dt_annulus,
                   m_dot,annulus_tpos_start,"little_m_dot")    
    

# set up variable array of M dot for every time position for every annulus
# (overall mass movement wihin an annulus.)

M_dot = M_dot(m_dot,disk_annuli,N_dt_annulus,dt_annulus,annulus_tpos_start) 

##del m_dot
##gc.collect()
#
#print_timestamp("Status: M_dot set up")
#
## plotting the M dot 
# 
#if (plot_all):
#    plot_mass_flow(disk_annuli,N_dt_annulus,dt_annulus,
#                   M_dot,annulus_tpos_start,"big_M_dot")  
#
## set up array of effective temperature (K) for every annulus
#
#disk_temperature = annulus_effective_temperature(M_dot,disk_annuli,N_dt_annulus,dt_annulus,annulus_tpos_start) 
#
#
#print_timestamp("Status: Temperature set up")
#
# # plot radius against temperature (not needed used as a check)
#
#if (plot_all):
#    plt.plot(disk_annuli, disk_temperature)
#    plt.xlabel("Disk Radius", fontsize =10)
#    plt.ylabel("Temperature", fontsize =10)
#    plt.savefig ('temperature.pdf')
#    plt.show()
#
#
## using all the data modelled on the disk above (temperature, radius, Mdot, mdot etc)
## we can now start using the model to make prediction eg flux from the disk. 
#
#if (plot_all):
#    plot_spectrum(disk_annuli,N_dt_annulus,dt_annulus, M_dot,annulus_tpos_start, 1000, disk_temperature)
#
#    print_timestamp("Status:Spectrum produced")
#
## create and plot lightcurve data (counts in each observation period against time)
#
##counts, time_axis = lightcurve(disk_annuli,N_dt_annulus,dt_annulus,
##                               M_dot,annulus_tpos_start, disk_temperature, 0.125e19)
#counts, time_axis = lightcurve(disk_annuli,N_dt_annulus,dt_annulus,
#                               M_dot,annulus_tpos_start, disk_temperature, 0.125e19)
##frequency       = 0.6e20 # x-ray
#    
#print_timestamp("Status: Lightcurve Produced")
#
#
## use periodogram method and return frequency and power density:
## let's put the sample frequency into physical units though...
#
#dtphys = Rt*(time_axis[1] - time_axis[0])
#samp_freq = 1.0/dtphys
#f, Pxx_den = signal.periodogram(counts,samp_freq)
#plt.semilogy(np.log10(f), Pxx_den) #power density already log
##plt.ylim([1e-7, 1e2])
#local_y_label = "PSD [V**2/Hz]_" + str(R_in) + "_" + str(M_body/M_sol)
##file = local_y_label + ".pdf"
#plt.xticks(np.arange(-6, 3, step=0.2),rotation=90)
#plt.ylabel(local_y_label, fontsize =10)
#plt.savefig ("PSD_" + str(R_in) + "_" + str(M_body/M_sol)+ ".pdf")
#plt.xlabel('log frequency [Hz]')
##plt.ylabel('PSD [V**2/Hz]')
##plt.savefig ('logsemilogy.pdf')
#plt.show()
#
#print_timestamp("array size = " + str(len(M_dot)))
#
Mdot0=M_dot[0:annulus_tpos_start[1]-1]/10000
Mdot1=M_dot[annulus_tpos_start[1]:annulus_tpos_start[2]-1]/10000
Mdot2=M_dot[annulus_tpos_start[2]:]/10000

time0=np.empty(len(Mdot0))
for t in range(len(Mdot0)):
    time0[t]=t*dt_annulus[0]
time1=np.empty(len(Mdot1))
for t in range(len(Mdot1)):
    time1[t]=t*dt_annulus[1]
time2=np.empty(len(Mdot2))
for t in range(len(Mdot2)):
    time2[t]=t*dt_annulus[2]

plt.plot(time0, Mdot0)
plt.plot(time1, Mdot1)
plt.plot(time2, Mdot2)
plt.xlabel("time", fontsize =10)
plt.ylabel("Mass Flow", fontsize =10)
plt.savefig ('M_overlay.pdf')
plt.show()