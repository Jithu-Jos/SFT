#!/usr/bin/env python
# coding: utf-8

# In[ ]:

## Importing required modules

import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
from tqdm import tqdm
import time as time

## End importing required modules


# ## Code Units

# In[ ]:


## The time in years for which the simulation is conducted
simul_time = 24.0     # 24 years

Rsun = 6.95e+10   # Solar radius

## Fundamental units 
T_unit = simul_time * 365.25 * 24.0 * 3600.0     # Time unit in seconds
B_unit = 10.0    # Magnetic field units in Gauss
L_unit =  Rsun  # Length units in centimeters 

## Derived units
V_unit = L_unit / T_unit    # Velocity unit in centimeters per second
# rho_unit = B_unit**2 / (8.0 * np.pi * V_unit**2)     # Density unit in kilograms per cubic centimeter

deg_2_rad = np.pi / 180.0


# ## Constants and Parameters

# In[ ]:


##solar radius
r = Rsun / L_unit   #(6.95e-10 cm)

## Diffusion constant
eta = 500e+10    # 500 cm2/s
eta = eta / (L_unit * V_unit)        

## Advection Velocity
u_0 = -12.5e+2     # Advection velocity in m/s
u_0 = u_0 / V_unit  
lam0 = 75.0*deg_2_rad  

## Decay Time
tau = 5.0 * 365.25 * 24.0 * 3600.0     # Decay time in seconds (5.0 years)
tau = tau / T_unit

## Initial amplitude of magnetic field
B0 = 10.0    # in Gauss
B0 = B0 / B_unit

# Source term parameters
ahat = 0.00185  
bhat = 48.7  
chat = 0.71
sourcescale = 0.003 * 10000 / B_unit 
cycleper = 11.0 * 365.25   # Solar cycle duration in days (11 years)


# ## Different terms of SFT

# In[ ]:


@jit(nopython = True)
def source_num(latitude, t):
    tc = 12.0 * ((((t / cycleper) % 1) * cycleper / 365.25))
    ampli = sourcescale * ahat * tc**3 / (np.exp(tc**2 / bhat**2) - chat)
    cycleno = int(t // cycleper) + 1
    evenodd = 1 - 2 * (cycleno % 2)
    lambda0 = 26.4 - 34.2 * ((t / cycleper) % 1) + 16.1 * ((t / cycleper) % 1)**2

    fwhm = 6.0
    joynorm = 0.5 / np.sin(20.0 / 180 * np.pi)

    bandn1 = evenodd * ampli * np.exp(-(latitude - lambda0 - joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)
    bandn2a = -evenodd * ampli * np.exp(-(latitude - lambda0 + joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)
    bands2a = evenodd * ampli * np.exp(-(latitude + lambda0 - joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)
    bands1 = -evenodd * ampli * np.exp(-(latitude + lambda0 + joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)

    Nfine = 180 + 1

    thetaf = np.linspace(0, np.pi, Nfine)
    latitudef = 90.0 - thetaf * 180 / np.pi

    bandn1f = evenodd * ampli * np.exp(-(latitudef - lambda0 - joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)
    bandn2af = -evenodd * ampli * np.exp(-(latitudef - lambda0 + joynorm * np.sin(lambda0 / 180 * np.pi))**2 / 2 / fwhm**2)

    integrand1 = -np.sin(thetaf) * bandn1f
    fluxband1 = np.trapz(integrand1, thetaf)

    integrand2 = -np.sin(thetaf) * bandn2af
    fluxband2 = np.trapz(integrand2, thetaf)

    fluxcorrection = 1.0
    if ampli != 0:
        fluxcorrection = -fluxband1 / fluxband2

    bandn2 = fluxcorrection * bandn2a
    bands2 = fluxcorrection * bands2a

    value = bandn1 + bandn2 + bands1 + bands2

    return value

# Calculate the source term 
@jit(nopython = True)
def source_numer(lam, t):
    return source_num(lam * 180 / np.pi, t * simul_time * 365.25)

# Advection velocity
@jit(nopython = True)
def vel(lam):
    return u_0 * np.sin(np.pi * lam / lam0)

# Define the advection term
@jit(nopython = True)
def adv(lam, t):
    return -(vel(lam) / r - eta * np.tan(lam) / r**2)

# Define the escape term 
@jit(nopython = True)
def esc(lam, r, eta, tau_esc):
    return (1.0 / tau_esc + vel(lam) * np.tan(lam) / r - eta * (1.0 / np.cos(lam)**2) / r**2)

# Define the initial condition
@jit(nopython = True)
def initial(lam):
    return B0 * np.sin(lam)


# In[ ]:


# Tridiagonal solver - Thomas algorithm
@jit(nopython = True)
def tridiag2(n, a, b, c, d):
    
    """
    Solve a tridiagonal system of equations of the form Ax = d,where A is a tridiagonal matrix,
    and x and d are vectors.
    
    Args:
        n: Number of equations (length of vectors).
        a: Lower diagonal of the matrix (length n-1).
        b: Main diagonal of the matrix (length n).
        c: Upper diagonal of the matrix (length n-1).
        d: Right-hand side vector (length n).
    
    Returns:
        The solution vector x.
    """
    
    
    E = np.zeros(n)
    F = np.zeros(n)
    new = np.zeros(n)
    
    for i in range(2,n-2):
        if i == 2:
            E[i] = -c[i] / b[i]
            F[i] = d[i] / b[i]
        else:
            E[i] = -c[i] / (a[i] * E[i-1] + b[i])
            F[i] = (d[i] - a[i] * F[i-1]) / (a[i] * E[i-1] + b[i])
    
    new[n-3] = F[n-3]
    
    for i in range(n-4, 1, -1):
        new[i] = E[i] * new[i+1] + F[i]
    
    new[res-2] = 0
    new[res-1] = 0
    new[0] = 0
    new[1] = 0
    return new


# ## Simulation domain

# In[ ]:


num_files = 1000    # Temporal resolution
Tmin = 0.0    # Initial time stamp
Tmax = 1.0    # Final time stamp

res = 1024  # 128,256,384,512,1024   Spatial resolution
lat_min = -70.0   # Minimum latitude in degrees
lat_max = 70.0   # Maximum latitude in degrees


# In[ ]:


# Discretization
dt = (Tmax - Tmin) / num_files  # Time step size
dlat = (lat_max - lat_min) / res  # Latitude step size
d_xi = np.copy(dlat*deg_2_rad)    # Latitude step size  (in degrees)

lat_left = np.zeros(res)  
lat_right = np.zeros(res)  
lat_cen = np.zeros(res)  
lat_width = np.zeros(res)  

for i in range(len(lat_cen)):
    lat_left[i] = (lat_min + i * dlat) * deg_2_rad  
    lat_right[i] = (lat_min + (i + 1) * dlat) * deg_2_rad  
    lat_cen[i] = (lat_right[i] + lat_left[i]) / 2.0  
    lat_width[i] = lat_right[i] - lat_left[i] 
#     print(lat_width[i])  


# In[ ]:


B_explicit=np.zeros((num_files+1,res))    # To store solution explicit scheme
B_RK_IMEX=np.zeros((num_files+1,res))    # To store solution RK-IMEX scheme


# In[ ]:


## initial condition

mag = np.zeros(res)
for i in range(len(mag)):
    mag[i] = initial(lat_cen[i])
    plt.plot(lat_cen[i], mag[i], '.')

plt.xlabel('Latitude')
plt.ylabel('Magnetic Field')
plt.title('Initial Magnetic Field Distribution')
plt.show()

B_explicit[0] = mag
B_RK_IMEX[0] = mag


# In[ ]:


courant_number = 0.4  # CFL number
courant_array = np.zeros(res)  

for i in range(len(lat_cen)):
    courant_array[i] = courant_number * lat_width[i] / np.abs(adv(lat_cen[i], 0.0))

dt_cfl = min(courant_array) 
dt_glb = dt  # Global time step

# print(dt_cfl, dt)

# Adjust time step if necessary based on Courant condition
if dt_cfl < dt:
    temp = dt / dt_cfl
    nsub = int(temp) + 1  # Number of substeps
    dt = dt / nsub  # Updated time step
else:
    nsub = 1
    dt = dt

print("steps =", nsub)
print("CFL dt =", dt)


# ## RK-IMEX

# In[ ]:


@jit(nopython = True)
def diffusion(rho, lat_width, k):
    return eta*(rho[k+1] - 2*rho[k] + rho[k-1])/(lat_width*lat_width)


@jit(nopython = True)
def advection(rho, grid_cent,grid_left, grid_right, dt, lat_width, i, t):
    if (adv(grid_right,t)>0.0):
        den1 = rho[i+1]-rho[i]
        den2 = rho[i]-rho[i-1]
        r = den1*den2/(den1 + den2)
        if(den1*den2>0.0):
            temp1 = adv(grid_right,t)*(rho[i] + r)
        else:
            temp1 = adv(grid_right,t)*rho[i]
            
    else:
        den1 = rho[i+1]-rho[i]
        den2 = rho[i+2]-rho[i+1]
        r = den1*den2/(den1 + den2)
        if(den1*den2>0.0):
            temp1 = adv(grid_right,t)*(rho[i+1] - r)
        else:
            temp1 = adv(grid_right,t)*rho[i+1]

    if (adv(grid_left,t)>0.0):
        den1 = rho[i]-rho[i-1]
        den2 = rho[i-1]-rho[i-2]
        r = den1*den2/(den1 + den2)
        if (den1*den2>0.0):
            temp2 = adv(grid_left,t)*(rho[i-1] + r)
        else:
            temp2 = adv(grid_left,t)*rho[i-1]
    else:
        den1 = rho[i+1]-rho[i]
        den2 = rho[i]-rho[i-1]
        r = den1*den2/(den1 + den2)
        if (den1*den2>0.0):
            temp2 = adv(grid_left,t)*(rho[i] - r)
        else:
            temp2 = adv(grid_left,t)*rho[i]

    u = (-(temp1-temp2)/lat_width)
    return u

"""
        Function used to update using RK-IMEX scheme, returns the updated value
        The function updates for each n_sub
        The function updates all the space points together
"""
@jit(nopython= True)
def rkimex(mag,time,dt):
    
    t = time
    # Coefficient arrays for tridiagonal solver in step 1
    A = np.zeros(res)  
    B = np.zeros(res)
    C = np.zeros(res)
    D = np.zeros(res)

    mag_new = np.zeros(res)
    mag_new1 = np.zeros(res)
    mag_new2 = np.zeros(res)

    gamma = (1.0-1.0/np.sqrt(2.0))

    ## Step 1  (Diffusion)
    for k in range (2,res-2):
        if (k==2):
            A[k] = 0.0
        else:
            A[k] = -gamma*eta*dt/(r**2*lat_width[k]**2)

        if (k==2):
            B[k] = 1.0 + gamma*eta*dt/(r**2*lat_width[k]**2) +gamma*esc(lat_cen[k],r,eta,tau)*dt
        elif(k==res-3):
            B[k] = 1.0 + gamma*eta*dt/(r**2*lat_width[k]**2) +gamma*esc(lat_cen[k],r,eta,tau)*dt
        else:
            B[k] = 1.0 + 2*gamma*eta*dt/(r**2*lat_width[k]**2) +gamma*esc(lat_cen[k],r,eta,tau)*dt

        if(k==res-3):
            C[k] = 0.0
        else:
            C[k] = -gamma*eta*dt/(r**2*lat_width[k]**2)

        D[k] = mag[k]  + gamma*source_numer(lat_cen[k],time)*dt


    mag_new = tridiag2(res,A,B,C,D)
    
    # Applying boundary conditions
    mag_new[0] = mag_new[3]
    mag_new[1] = mag_new[2]
    mag_new[res-2] = mag_new[res-3]
    mag_new[res-1] = mag_new[res-4]
    
    
    # Coefficient arrays for tridiagonal solver in step 2
    A = np.zeros(res)  
    B = np.zeros(res)
    C = np.zeros(res)
    D = np.zeros(res)

    ## Step 2 (Diffusion and Advection)
    for k in range (2,res-2):
        if (k==2):
            A[k] = 0.0
        else:
            A[k] = -gamma*eta*dt/(r**2*lat_width[k]**2)

        if (k==2):
            B[k] = 1.0 + gamma*eta*dt/(r**2*lat_width[k]**2)+gamma*esc(lat_cen[k],r,eta,tau)*dt
        elif(k==res-3):
            B[k] = 1.0 + gamma*eta*dt/(r**2*lat_width[k]**2)+gamma*esc(lat_cen[k],r,eta,tau)*dt
        else:
            B[k] = 1.0 + 2*gamma*eta*dt/(r**2*lat_width[k]**2)+gamma*esc(lat_cen[k],r,eta,tau)*dt

        if(k==res-3):
            C[k] = 0.0
        else:
            C[k] = -gamma*eta*dt/(r**2*lat_width[k]**2)
            
        D[k] = mag[k]  + source_numer(lat_cen[k],time)*dt

        D[k] = D[k] + dt*((1-2*gamma)*diffusion(mag_new, d_xi, k))
        

        
        D[k] = D[k] + dt*(advection(mag_new, lat_cen, lat_left[k],
                                                       lat_right[k], dt, d_xi, k,t))

        D[k] = D[k] - dt*((1-2*gamma)*mag_new[k]*esc(lat_cen[k], r, eta, tau))
        D[k] = D[k] + (1-2*gamma)*source_numer(lat_cen[j],t)*dt
        
    mag_new1 = tridiag2(res,A,B,C,D)

    # Applying boundary conditions
    mag_new1[0] = mag_new1[3]
    mag_new1[1] = mag_new1[2]
    mag_new1[res-2] = mag_new1[res-3]
    mag_new1[res-1] = mag_new1[res-4]
    

    # Step 3 (Diffusion and Advection)
    for k in range(2, res-2):
    
        mag_new2[k] = mag[k] 
        mag_new2[k] = mag_new2[k] + dt*(diffusion(mag_new, d_xi, k)+ diffusion(mag_new1, d_xi, k))/2.0

        mag_new2[k] = mag_new2[k] + dt*(
            advection(mag_new, lat_cen, lat_left[k], lat_right[k],
                      dt, d_xi, k, time)
        + advection(mag_new1, lat_cen, lat_left[k], lat_right[k],
                      dt, d_xi, k, time))/2.0

        mag_new2[k] = mag_new2[k] - dt*((mag_new[k]*esc(lat_cen[k], r, eta, tau)) +
                                      (mag_new1[k]*esc(lat_cen[k], r, eta, tau)))/2.0

        mag_new2[k] = mag_new2[k] + source_numer(lat_cen[k],t)*dt

    # Applying boundary conditions
    mag_new2[0] = mag_new2[3]
    mag_new2[1] = mag_new2[2]
    mag_new2[res-2] = mag_new2[res-3]
    mag_new2[res-1] = mag_new2[res-4]

    return mag_new2


# ## Explicit scheme

# In[ ]:


"""
        Function used to update using standard scheme  returns the updated value
        The function updates for 1 n_sub
        The function updates all the space points together
"""


@jit(nopython= True)
def lat_update(mag,time,dt):
    t = time
    
    # Coefficient arrays for tridiagonal solver
    A = np.zeros(res)  
    B = np.zeros(res)
    C = np.zeros(res)
    D = np.zeros(res)
    
    mag_new = np.zeros(res)
    
    # Upwind scheme calculation for magnetic field advection
    for k in range(2, res-2):
        right = adv(lat_right[k], time)  # Right-sided advection velocity
        left = adv(lat_left[k], time)  # Left-sided advection velocity
        left_half = np.zeros(res-2)
        right_half = np.zeros(res-2)
        
        for k1 in range(1, res-2):
            if (mag[k1+1] - mag[k1]) * (mag[k1] - mag[k1-1]) > 0:
                left_half[k1] = mag[k1] + (mag[k1+1] - mag[k1]) * (mag[k1] - mag[k1-1]) / ((mag[k1+1] - mag[k1]) + (mag[k1] - mag[k1-1]))
            else:
                left_half[k1] = mag[k1]
                
            if (mag[k1+2] - mag[k1+1]) * (mag[k1+1] - mag[k1]) > 0:
                right_half[k1] = mag[k1+1] - (mag[k1+2] - mag[k1+1]) * (mag[k1+1] - mag[k1]) / ((mag[k1+2] - mag[k1+1]) + (mag[k1+1] - mag[k1]))
            else:
                right_half[k1] = mag[k1+1]
                    
        # Apply upwind scheme based on advection velocities
        if right >= 0 and left >= 0:
            mag_new[k] = mag[k] - dt * (right * left_half[k] - left * left_half[k-1]) / lat_width[k]
        elif right >= 0 and left < 0:
            mag_new[k] = mag[k] - dt * (right * left_half[k] - left * right_half[k-1]) / lat_width[k]
        elif right < 0 and left >= 0:
            mag_new[k] = mag[k] - dt * (right * right_half[k] - left * left_half[k-1]) / lat_width[k]
        elif right < 0 and left < 0:
            mag_new[k] = mag[k] - dt*(right*right_half[k]-left*right_half[k-1])/lat_width[k]

    # Applying boundary conditions
    mag_new[0] = mag[3]
    mag_new[1] = mag[2]
    mag_new[res-2] = mag[res-3]
    mag_new[res-1] = mag[res-4]
    
    for k in range(res):
        mag[k] = mag_new[k]
            
    ## Diffusion and decay terms solver
    for k in range (res):
        if (k==2):
            A[k] = 0.0
        else:
            A[k] = -eta*dt/(r**2*lat_width[k]**2)

        if (k==2):
            B[k] = 1.0 + eta*dt/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt
        elif(k==res-3):
            B[k] = 1.0 + eta*dt/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt
        else:
            B[k] = 1.0 + 2*eta*dt/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt

        if(k==res-3):
            C[k] = 0.0
        else:
            C[k] = -eta*dt/(r**2*lat_width[k]**2)

        D[k] = mag_new[k]  + source_numer(lat_cen[k],time)*dt
        
    mag = tridiag2(res,A,B,C,D)
    
    # Applying boundary conditions
    mag[0] = mag[3]
    mag[1] = mag[2]
    mag[res-2] = mag[res-3]
    mag[res-1] = mag[res-4]
    return mag


# ## Solving

# In[ ]:


# update using Explicit scheme
time = np.zeros(num_files+1) 
time[0] = 0.0  
for i in tqdm(range(num_files)):
    time[i+1] = time[i] + dt_glb  # Calculate current time step
    for j in range(nsub):
        mag_1 = lat_update(mag,time[i]+j*dt,dt)
        K1 = mag_1 - mag
        mag_2 = lat_update(mag_1,time[i]+(j+1)*dt,dt)
        K2 = mag_2 - mag_1
        mag = mag + 0.5*(K1+K2)
    B_explicit[i+1] = mag

# update using RK-IMEX scheme   
mag = np.copy(B_RK_IMEX[0])
time = np.zeros(num_files+1)  
time[0] = 0.0  
for i in tqdm(range(num_files)):
    time[i+1] = time[i] + dt_glb
    for j in range(nsub):
        mag_1 = rkimex(mag,time[i]+j*dt,dt)
        mag = np.copy(mag_1)
    B_RK_IMEX[i+1] = mag_1


# In[ ]:


## To save the files
# variable_name = f"numeric_1D_num_{res}.npy"
# np.save(variable_name, B_explicit)

# variable_name = f"numeric_1D_rk_{res}.npy"
# np.save(variable_name, B_RK_IMEX)


# ## Plotting

# In[ ]:


im = plt.imshow(B_explicit.T*10.0, vmin=-5.0, vmax=5.0, aspect="auto", origin="lower", extent=[0, simul_time, lat_min, lat_max])
plt.xlabel("Time")
plt.ylabel("Latitude")
plt.title("PINNs")
plt.colorbar(im,location="bottom", aspect=70, extend="both", label="B")
plt.title("Magnetic field solving the 1D SFT equation   num")
plt.show()


im = plt.imshow(B_RK_IMEX.T*10.0, vmin=-5.0, vmax=5.0, aspect="auto", origin="lower", extent=[0, simul_time, lat_min, lat_max])
plt.xlabel("Time")
plt.ylabel("Latitude")
plt.title("PINNs")
plt.colorbar(im,location="bottom", aspect=70, extend="both", label="B")
plt.title("Magnetic field solving the 1D SFT equation  rk")
plt.show()

