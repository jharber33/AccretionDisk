
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
'>>-------------------------------To-Do-List--------------------------------<<'
"""
Make time depend on viscous time
MAX time - visc outer
MIN time aka timesteps aka sample rate - visc inner
Replace global variables with local variables 
"""
#----------------------------------------------------------------------------
'>>--------------------------------Constants--------------------------------<<'

c = 2.99792458e8    # Speed of Light [ metres per second ]
G = 6.67259e-11     # Gravitational Constant  [ MKS units ] 
sigma = 5.6703e-8   #Stefan-Boltzmann Constant [Watt m^-2 K^-4]

#----------------------------------------------------------------------------
'>>--------------------------------Variables--------------------------------<<'

R_in   = 5.0   # Radius of Central Body, eg. Black Hole, also the start of the disk 
M_body = 1.0   # Mass of Central Body
R_out  = 25.0  # Radial distance to edge of disk
N      = 100    # Number of Annuli , N > 1 at all times 
H      = 1.0   # Disk Height / Thickness
M0_dot = 1.0   # Accretion Rate at outer radii, where mass first flows in 
alpha  = 2000  # Viscosity Parameter 
sf     = 0.1   # Scale Factor, m_dot << 1
Time   = 1000  # Time to Run Simulation

#----------------------------------------------------------------------------
'>>--------------------------------Functions--------------------------------<<'


def Accretion_Disk (N, R_in, R_out):
    """
    Annuli = array of all the annulus (defined by calling accretion disk array so annuli = R)
        Function "accretion_disk_array" to split the disk into N annuli     
        This function creates an empty disk of multiple annuli to be later filled
        Annuli are linearly distributed across the disk 

    Inputs: Number of Annuli, inner radius, outer radius
    Output: Array R of the radii of tha Annuli
    """
    R = np.empty(N)
    ratio = (R_out/R_in) ** (1.0/(N-1))
    for i in range(N):
        #R[i] = (((R_out - R_in) / N) * i) + R_in   # linear progression
        R[i] = R_in * (ratio ** i)                 # geometric progression
    return R

annuli= Accretion_Disk(N, R_in, R_out) #Temporary Global Variable


def Viscous_Timescale(R):    
    """
    Function "viscous_timescale" to calculate the viscous velecity at a particular radius 
    Using the formula given in (enter ref)
    Inputs: Radius
    Output: viscous velecity at that radius
    """
    t_viscous = (2*np.pi)*R**(3.0/2.0) / ((H/R_out)**(2.0) * alpha)
    return t_viscous  


def Phase_Shift(annulus):
    """
    Function "Phase_Shift" to calculate the delta viscous velecity at a given annulus 
    Inputs: Annulus
    Output: Delta Timescale
    """
    delta_t = Viscous_Timescale(annuli[annulus+1]) - Viscous_Timescale(annuli[annulus])
    return int(delta_t)


def Viscous_Frequency(R):
    """
    This creates an individual frequency value for each annulus
    """
    t_visc = Viscous_Timescale(R)
    f_visc = 1 / t_visc
    return f_visc



def m_dot():
    """
    Function: "m_dot" to calculate variable rate of change of mass in a particular annulus 
        For each annulus calculate the radius and viscous timescalethen 
        for each time division use the timescale as the frequency of a sine wave
        calculate the small periodic changes in each annuli
        
    Inputs: None as it cycles through each annulus
    Output: Array of lowercase m_dot changes
    """
    
    m_dot = np.empty((len(annuli),Time))
    for annulus in range(N):
        
        r=annuli[annulus] #particular annulus in the annuli array 0=inner radius
        timescale = (Viscous_Timescale(r) / (2.0 * np.pi)) #int to try and line up waves as timescale should be a
        for t in range(Time):
            m_dot[annulus][t] = sf * np.sin(t / timescale)
    return m_dot

m_dot=m_dot() #Temporary Global Variable

def M_dot():
    """
    Function: "M_dot" to calculate the overall rate of change of mass in a particular annulus 
        For inner annului the rate of flow is determined by the rate from the next annuli and the small change m_dot   
        the rate from the outer annuli is time delayed by the time it takes the mass to cross that annuli  (t_offset)
        The code cycles through the annuli from the outside (penultimate annulus) to the inside
        As the pattern is cyclical I assume that is the time is smaller than the offset we can start the cycle again
    
    Inputs: None as it cycles through each annulus
    Output: Array of Capital M_dot changes
    """
    M_dot = np.empty((len(annuli),Time))
    #For outer annulus there is no other input than the base rate of flow into the disk (M_0)and the small change in that annuli
    for t in range(Time):
        M_dot[(len(annuli)-1)][t] = M0_dot * (1.0 + m_dot[(len(annuli)-1)][t])
   
    for i in range(len(annuli)-2,-1,-1): #backwards through array 
        t_offset_annulus = Phase_Shift(i)
        for t in range(Time):
            t_offset = t - t_offset_annulus 
            if t_offset < 0:
                t_offset = t_offset + Time
            M_dot[i][t]=M_dot[i+1][t_offset] *(1.0 + m_dot[i][t]) 
    return M_dot

M_dot=M_dot() #Temporary Global Variable



def Viscous_Velocity(R): 
    """
    Radial Drift Velocity
    """
    v_visc = R**(-1.0/2.0) * (H/R_out)**(2.0) * alpha
    return v_visc

def Emissivity_Profile(R):
    """
    Emissivity Profile for use in radiation of heat and light 
    """
    E = R**(-3.0)*(1 - (R_in/R)**(1.0/2.0))
    return E


def Effective_Temperature(R): 
    """
    Effective Temperature, used in deteremining the blackbody spectrum
    """
    T_eff = ((3 * G * m_dot * M_body)/(8 * np.pi * (R**3.0) * sigma))**(1.0/4.0)
    return T_eff

#----------------------------------------------------------------------------
'>>--------------------------------Plotting-------------------------------<<'

y0_plot=np.empty(Time)
y1_plot=np.empty(Time)
x_plot=np.empty(Time)

for i in range(Time):
    x_plot[i]=i
    y0_plot[i]=M_dot[0][i]
    y1_plot[i]=M_dot[1][i] 
    
    
plt.figure(figsize = (8,5))
plt.title("Mass Accretion Rate for annulus 0, 1", fontsize =15)
plt.xlabel("x", fontsize =10)
plt.ylabel("y", fontsize =10)
plt.plot(x_plot,y0_plot,'b',linewidth=1, label ='$Cˆ{12}$')
plt.plot(x_plot,y1_plot,'r',linewidth=1, label ='$Cˆ{12}$' )
plt.show()
