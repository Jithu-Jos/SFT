## Importing required modules

import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
from tqdm import tqdm
import time as time

## End importing required modules

# ## Code Units

## The time in years for which the simulation is conducted
simul_time = 0.30     # in years

Rsun = 6.95e+10   # Solar radius

## Fundamental units 
T_unit = simul_time * 365.25 * 24.0 * 3600.0     # Time unit in seconds
B_unit = 10.0    # Magnetic field units in Gauss
L_unit =  Rsun  # Length units in centimeters 

## Derived units
V_unit = L_unit / T_unit    # Velocity unit in centimeters per second
# rho_unit = B_unit**2 / (8.0 * np.pi * V_unit**2)     # Density unit in kilograms per cubic centimeter

deg_2_rad = np.pi / 180.0

# ## Constants and parameters

##solar radius
r = Rsun / L_unit   # (6.95e-10 cm)

## Diffusion constant
eta = 500e+10    #500km2/s      
eta = eta/(L_unit * V_unit)        

## Advection Velocity
u_0 = -12.5e+2     #12.5m/s      
u_0 = u_0/V_unit                     
lam0 = 75.0*deg_2_rad

## Initial amplitude of magnetic field
B0 = 10.0/B_unit    # 10 Gauss

## Decay Time
tau = 5.0*365.25*24.0*3600.0     #5.0 years
tau = tau/T_unit

# Initial condition
def initial(lam, phi):

    theta = np.pi/2 - lam    
    Bmax = 1.0  
    b = 0.4  
    tilt = 15.0*deg_2_rad
    theta0 = np.pi/2.0

    xphi =  2.0*deg_2_rad/(np.sin(theta0)*np.tan(tilt))
    
    phi_minus = 1.5*np.pi + xphi
    phi_plus = 1.5*np.pi - xphi
    theta_plus = (np.pi / 2) - 1.0*deg_2_rad - 15.0*deg_2_rad
    theta_minus = (np.pi / 2) + 1.0*deg_2_rad - 15.0*deg_2_rad

    cos_theta_plus = np.cos(theta_plus)
    cos_theta_minus = np.cos(theta_minus)
    sin_theta_plus = np.sin(theta_plus)
    sin_theta_minus = np.sin(theta_minus)

    cos_beta_plus = (cos_theta_plus * np.cos(theta) +
                     sin_theta_plus * np.sin(theta) * np.cos(phi - phi_plus))

    cos_beta_minus = (cos_theta_minus * np.cos(theta) +
                      sin_theta_minus * np.sin(theta) * np.cos(phi - phi_minus))

    cos_rho_0 = (cos_theta_plus * cos_theta_minus +
                 sin_theta_plus * sin_theta_minus * np.cos(phi_plus - phi_minus))

    delta = b * np.arccos(cos_rho_0)

    B_plus = Bmax * np.exp(-2 * (1 - cos_beta_plus) / delta**2)
    B_minus = Bmax * np.exp(-2 * (1 - cos_beta_minus) / delta**2)

    delta_Br = B_plus - B_minus
    return delta_Br
    
print("eta",eta)
print("tau",tau)
print("u_0",u_0)

# ## Different terms of SFT

# Advection velocity
@jit(nopython=True)
def vel(lam):
    return u_0 * np.sin(np.pi * lam / lam0)

# Define the advection term
@jit(nopython=True)
def adv_lat(lam, r, eta):
    return -(vel(lam) / r - eta * np.tan(lam) / r**2)

#  Differential rotation
@jit(nopython=True)
def adv_lon(lam):
    th = np.pi/2 - lam
    omega = 0.18 - 2.36 * np.cos(th)**2 - 1.787 * np.cos(th)**4 
    omega = omega*np.pi/180
    return omega*T_unit/(24.0*60*60)

# Source term
@jit(nopython=True)
def source_numer(a,b):
    return 0.0

# Define the longitude diffusion term
@jit(nopython=True)
def diff_lon(eta,lam):
    return eta/(np.cos(lam)**2)

# Define the escape term 
@jit(nopython=True)
def esc(lam, r, eta, tau_esc):
    return (1.0 / tau_esc + vel(lam) * np.tan(lam) / r - eta * (1.0 / np.cos(lam)**2) / r**2)

# Tridiagonal solver - Thomas algorithm
@jit(nopython=True)
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
    
    for i in range(n):
        if i == 2:
            E[i] = -c[i] / b[i]
            F[i] = d[i] / b[i]
        else:
            E[i] = -c[i] / (a[i] * E[i-1] + b[i])
            F[i] = (d[i] - a[i] * F[i-1]) / (a[i] * E[i-1] + b[i])
    
    new[n-3] = F[n-3]
    
    for i in range(n-4, 1, -1):
        new[i] = E[i] * new[i+1] + F[i]
    
    new[n-2] = 0
    new[n-1] = 0
    new[0] = 0
    new[1] = 0
    return new
    
# # Simulation domain

lat_max = 0.40*180.0  # Maximum latitude in degrees
lat_min = -0.40*180.0 # Minimum latitude in degrees
res_lat = 72    #128,256,512,1024   #Resolution of latitude

lon_max = 360.0  # Maximum longitude in degrees
lon_min = 0.0 # Minimum longitude in degrees
res_lon = 180   #128,256,512,1024   #Resolution of longitude


Tmin = 0.0   # Initial time stamp
Tmax = 1.0 # Final time stamp
num_files = 1000    #Number of time stamps for which the result is produced
dt = (Tmax - Tmin) / num_files  # Time step size

# Latitude discretization
dlat = (lat_max - lat_min) / res_lat  # Latitude step size

lat_left = np.zeros(res_lat)  
lat_right = np.zeros(res_lat)  
lat_cen = np.zeros(res_lat)  
lat_width = np.zeros(res_lat)  

for i in range(len(lat_cen)):
    lat_left[i] = (lat_min + i * dlat) * deg_2_rad
    lat_right[i] = (lat_min + (i + 1) * dlat) * deg_2_rad
    lat_cen[i] = (lat_right[i] + lat_left[i]) / 2.0
    lat_width[i] = lat_right[i] - lat_left[i]
#     print(lat_width[i])  

# Longitude discretization
dlon = (lon_max - lon_min) / res_lon  # Latitude step size

lon_left = np.zeros(res_lon)
lon_right = np.zeros(res_lon)
lon_cen = np.zeros(res_lon)
lon_width = np.zeros(res_lon)

for i in range(len(lon_cen)):
    lon_left[i] = (lon_min + i * dlon) * deg_2_rad
    lon_right[i] = (lon_min + (i + 1) * dlon) * deg_2_rad
    lon_cen[i] = (lon_right[i] + lon_left[i]) / 2.0
    lon_width[i] = lon_right[i] - lon_left[i]
#     print(lon_width[i])  

B_Explicit = np.zeros((num_files+1,res_lat,res_lon)) 
B_RK_IMEX = np.zeros((num_files+1,res_lat,res_lon)) 

## initial condition

mag = np.zeros((res_lat,res_lon))
for i in range(len(mag)):
    for j in range(len(mag[0])):
        mag[i,j] = initial(lat_cen[i],lon_cen[j])

plt.imshow(mag, origin="lower")
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Initial Magnetic Field Distribution')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection = "hammer")
LAC,LOC =np.meshgrid(lat_cen,lon_cen - np.pi)
im = plt.pcolormesh(LOC,LAC,mag.T,cmap=plt.cm.bwr,shading='gouraud')#,vmax=0.05,vmin = -0.05)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.axhline(y=0,color = "k", linestyle = ":")
plt.show()

B_Explicit[0] = mag
B_RK_IMEX[0] = mag

courant_number = 0.4  # CFL number
courant_array = np.zeros(res_lat)

for i in range(len(lat_cen)):
    courant_array[i] = courant_number * lat_width[i] / np.abs(adv_lat(lat_cen[i], r, eta))

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

courant_array_lon = np.zeros((res_lon,res_lat))  

for i in range(len(lon_cen)):
    for j in range(len(lat_cen)):
        courant_array_lon[i,j] = courant_number * lon_width[i] / np.abs(adv_lon(lat_cen[j]))

dt_cfl = min(courant_array_lon.flatten())

print(dt_cfl, dt)

# Adjust time step if necessary based on Courant condition
if dt_cfl < dt:
    temp = dt / dt_cfl
    nsub = int(temp) + 1  # Number of substeps
    dt = dt / nsub  # Updated time step
else:
    nsub = nsub
    dt = dt

print("CFL dt =", dt)
print("steps =", nsub)

# ## Explicit Scheme

"""
        Function used to update latitude using explicit scheme, returns the updated value
        The function updates for each n_sub
        The function updates all longitude together
"""
@jit(nopython=True)
def lat_update(mag, nsub, dt, dt_glb):
    for lon in range(res_lon):
        # Coefficient arrays for tridiagonal solver
        A = np.zeros(res_lat)  
        B = np.zeros(res_lat)
        C = np.zeros(res_lat)
        D = np.zeros(res_lat)
        mag_new = np.zeros((res_lat,res_lon))
        # Upwind scheme calculation for magnetic field advection
        for k in range(2, res_lat-2):
            right = adv_lat(lat_right[k], r, eta)  # Right-sided advection velocity
            left = adv_lat(lat_left[k], r, eta)   # Left-sided advection velocity
            left_half = np.zeros(res_lat-2)
            right_half = np.zeros(res_lat-2)

            for k1 in range(1, res_lat-2):
                if (mag[k1+1,lon] - mag[k1,lon]) * (mag[k1,lon] - mag[k1-1,lon]) > 0:
                    left_half[k1] = mag[k1,lon] + (mag[k1+1,lon] - mag[k1,lon]) * (mag[k1,lon] - mag[k1-1,lon]) / ((mag[k1+1,lon] - mag[k1,lon]) + (mag[k1,lon] - mag[k1-1,lon]))
                else:
                    left_half[k1] = mag[k1,lon]

                if (mag[k1+2,lon] - mag[k1+1,lon]) * (mag[k1+1,lon] - mag[k1,lon]) > 0:
                    right_half[k1] = mag[k1+1,lon] - (mag[k1+2,lon] - mag[k1+1,lon]) * (mag[k1+1,lon] - mag[k1,lon]) / ((mag[k1+2,lon] - mag[k1+1,lon]) + (mag[k1+1,lon] - mag[k1,lon]))
                else:
                    right_half[k1] = mag[k1+1,lon]

            # Apply upwind scheme based on advection velocities
            if right >= 0 and left >= 0:
                mag_new[k,lon] = mag[k,lon] - dt * (right * left_half[k] - left * left_half[k-1]) / lat_width[k]
            elif right >= 0 and left < 0:
                mag_new[k,lon] = mag[k,lon] - dt * (right * left_half[k] - left * right_half[k-1]) / lat_width[k]
            elif right < 0 and left >= 0:
                mag_new[k,lon] = mag[k,lon] - dt * (right * right_half[k] - left * left_half[k-1]) / lat_width[k]
            elif right < 0 and left < 0:
                mag_new[k,lon] = mag[k,lon] - dt*(right*right_half[k]-left*right_half[k-1])/lat_width[k]

        # Applying boundary conditions
        mag_new[0,lon] = mag[3,lon]
        mag_new[1,lon] = mag[2,lon]
        mag_new[res_lat-2,lon] = mag[res_lat-3,lon]
        mag_new[res_lat-1,lon] = mag[res_lat-4,lon]
        
        for k in range(res_lat):
            mag[k,lon] = mag_new[k,lon]
            
        ## Diffusion and decay terms solver
        for k in range (res_lat):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -eta*dt_glb/(r**2*lat_width[k]**2)

            if (k==2):
                B[k] = 1.0+eta*dt_glb/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt_glb
            elif(k==res_lat-3):
                B[k] = 1.0+eta*dt_glb/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt_glb
            else:
                B[k] = 1.0+2*eta*dt_glb/(r**2*lat_width[k]**2)+esc(lat_cen[k],r,eta,tau)*dt_glb

            if(k==res_lat-3):
                C[k] = 0.0
            else:
                C[k] = -eta*dt_glb/(r**2*lat_width[k]**2)

            D[k] = mag_new[k,lon]

        mag[:,lon] = tridiag2(res_lat,A,B,C,D)

        # Applying boundary conditions
        mag[0,lon] = mag[3,lon]
        mag[1,lon] = mag[2,lon]
        mag[res_lat-2,lon] = mag[res_lat-3,lon]
        mag[res_lat-1,lon] = mag[res_lat-4,lon]
        
    return mag

"""
        Function used to update longitude using explicit scheme, returns the updated value
        The function updates for each n_sub
        The function updates all latitude together
"""
@jit(nopython=True)
def lon_update(mag, nsub, dt, dt_glb):
    for lat in range(res_lat):
        # Coefficient arrays for tridiagonal solver
        A = np.zeros(res_lon) 
        B = np.zeros(res_lon)
        C = np.zeros(res_lon)
        D = np.zeros(res_lon)
        mag_new = np.zeros((res_lat,res_lon))
        # Upwind scheme calculation for magnetic field advection
        for k in range(2, res_lon-2):
            right = adv_lon(lat_cen[lat])  # Right-sided advection velocity
            left = adv_lon(lat_cen[lat])   # Left-sided advection velocity
            left_half = np.zeros(res_lon-2)
            right_half = np.zeros(res_lon-2)

            for k1 in range(1, res_lon-2):
                if (mag[lat,k1+1] - mag[lat,k1]) * (mag[lat,k1] - mag[lat,k1-1]) > 0:
                    left_half[k1] = mag[lat,k1] + (mag[lat,k1+1] - mag[lat,k1]) * (mag[lat,k1] - mag[lat,k1-1]) / ((mag[lat,k1+1] - mag[lat,k1]) + (mag[lat,k1] - mag[lat,k1-1]))
                else:
                    left_half[k1] = mag[lat,k1]

                if (mag[lat,k1+2] - mag[lat,k1+1]) * (mag[lat,k1+1] - mag[lat,k1]) > 0:
                    right_half[k1] = mag[lat,k1+1] - (mag[lat,k1+2] - mag[lat,k1+1]) * (mag[lat,k1+1] - mag[lat,k1]) / ((mag[lat,k1+2] - mag[lat,k1+1]) + (mag[lat,k1+1] - mag[lat,k1]))
                else:
                    right_half[k1] = mag[lat,k1+1]

            # Apply upwind scheme based on advection velocities
            if right >= 0 and left >= 0:
                mag_new[lat,k] = mag[lat,k] - dt * (right * left_half[k] - left * left_half[k-1]) / lon_width[k]
            elif right >= 0 and left < 0:
                mag_new[lat,k] = mag[lat,k] - dt * (right * left_half[k] - left * right_half[k-1]) / lon_width[k]
            elif right < 0 and left >= 0:
                mag_new[lat,k] = mag[lat,k] - dt * (right * right_half[k] - left * left_half[k-1]) / lon_width[k]
            elif right < 0 and left < 0:
                mag_new[lat,k] = mag[lat,k] - dt*(right*right_half[k]-left*right_half[k-1])/lon_width[k]

        # Applying boundary conditions
        mag_new[lat,0] = mag[lat,res_lon-4]
        mag_new[lat,1] = mag[lat,res_lon-3]
        mag_new[lat,res_lon-2] = mag[lat,2]
        mag_new[lat,res_lon-1] = mag[lat,3]

        for k in range(res_lon):
            mag[lat,k] = mag_new[lat,k]

        # Diffusion terms solver
        for k in range (res_lon):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -diff_lon(eta,lat_cen[lat])*dt_glb/(r**2*lon_width[k]**2)

            if (k==2):
                B[k] = 1.0 + diff_lon(eta,lat_cen[lat])*dt_glb/(r**2*lon_width[k]**2)
            elif(k==res_lat-3):
                B[k] = 1.0 + diff_lon(eta,lat_cen[lat])*dt_glb/(r**2*lon_width[k]**2)
            else:
                B[k] = 1.0 + 2*diff_lon(eta,lat_cen[lat])*dt_glb/(r**2*lon_width[k]**2)

            if(k==res_lat-3):
                C[k] = 0.0
            else:
                C[k] = -diff_lon(eta,lat_cen[lat])*dt_glb/(r**2*lon_width[k]**2)

            D[k] = mag_new[lat,k]

        mag[lat,:] = tridiag2(res_lon,A,B,C,D)

        # Applying boundary conditions
        mag_new[lat,0] = mag[lat,res_lon-4]
        mag_new[lat,1] = mag[lat,res_lon-3]
        mag_new[lat,res_lon-2] = mag[lat,2]
        mag_new[lat,res_lon-1] = mag[lat,3]

    return mag

# ## RK-IMEX

# Diffusion for RK-IMEX
@jit(nopython = True)
def diffusion(rho, dx, k, lat=0, lon=0, q=0):
    if lat==1:
        return diff_lat(eta,lon_cen[q])*(rho[k+1] - 2*rho[k] + rho[k-1])/(dx*dx)
    elif lon==1:
        return diff_lon(eta,lat_cen[q])*(rho[k+1] - 2*rho[k] + rho[k-1])/(dx*dx)

# Advection term for RK-IMEX
@jit(nopython = True)
def advection(rho, grid_cent,grid_left, grid_right, dt, dx, i, t, lat=0, lon=0):
    if lat==1:
        if (adv_lat(grid_right)>0.0):
            den1 = rho[i+1]-rho[i]
            den2 = rho[i]-rho[i-1]
            r = den1*den2/(den1 + den2)
            if(den1*den2>0.0):
                temp1 = adv_lat(grid_right)*(rho[i] + r)
            else:
                temp1 = adv_lat(grid_right)*rho[i]

        else:
            den1 = rho[i+1]-rho[i]
            den2 = rho[i+2]-rho[i+1]
            r = den1*den2/(den1 + den2)
            if(den1*den2>0.0):
                temp1 = adv_lat(grid_right)*(rho[i+1] - r)
            else:
                temp1 = adv_lat(grid_right)*rho[i+1]

        if (adv_lat(grid_left)>0.0):
            den1 = rho[i]-rho[i-1]
            den2 = rho[i-1]-rho[i-2]
            r = den1*den2/(den1 + den2)
            if (den1*den2>0.0):
                temp2 = adv_lat(grid_left)*(rho[i-1] + r)
            else:
                temp2 = adv_lat(grid_left)*rho[i-1]
        else:
            den1 = rho[i+1]-rho[i]
            den2 = rho[i]-rho[i-1]
            r = den1*den2/(den1 + den2)
            if (den1*den2>0.0):
                temp2 = adv_lat(grid_left)*(rho[i] - r)
            else:
                temp2 = adv_lat(grid_left)*rho[i]

        u = (-(temp1-temp2)/dx)
        return u
    
    elif lon==1:
        if (adv_lon(grid_right)>0.0):
            den1 = rho[i+1]-rho[i]
            den2 = rho[i]-rho[i-1]
            r = den1*den2/(den1 + den2)
            if(den1*den2>0.0):
                temp1 = adv_lon(grid_right)*(rho[i] + r)
            else:
                temp1 = adv_lon(grid_right)*rho[i]

        else:
            den1 = rho[i+1]-rho[i]
            den2 = rho[i+2]-rho[i+1]
            r = den1*den2/(den1 + den2)
            if(den1*den2>0.0):
                temp1 = adv_lon(grid_right)*(rho[i+1] - r)
            else:
                temp1 = adv_lon(grid_right)*rho[i+1]

        if (adv_lon(grid_left)>0.0):
            den1 = rho[i]-rho[i-1]
            den2 = rho[i-1]-rho[i-2]
            r = den1*den2/(den1 + den2)
            if (den1*den2>0.0):
                temp2 = adv_lon(grid_left)*(rho[i-1] + r)
            else:
                temp2 = adv_lon(grid_left)*rho[i-1]
        else:
            den1 = rho[i+1]-rho[i]
            den2 = rho[i]-rho[i-1]
            r = den1*den2/(den1 + den2)
            if (den1*den2>0.0):
                temp2 = adv_lon(grid_left)*(rho[i] - r)
            else:
                temp2 = adv_lon(grid_left)*rho[i]

        u = (-(temp1-temp2)/dx)
        return u

"""
        Function used to update in longitude using RK-IMEX scheme, returns the updated value
        The function updates for each n_sub
        The function updates all latitude together
"""

@jit(nopython= True)
def rkimex_lon(mag,time,dt):
    plt.imshow(mag)
    plt.colorbar()
    plt.show()
    for lat in range(res_lat):
        t = time
        # Coefficient arrays for tridiagonal solver for step 1
        A = np.zeros(res_lon)  
        B = np.zeros(res_lon)
        C = np.zeros(res_lon)
        D = np.zeros(res_lon)

        mag_new = np.zeros((res_lat,res_lon))
        mag_new1 = np.zeros((res_lat,res_lon))
        mag_new2 = np.zeros((res_lat,res_lon))

        gamma = (1.0-1.0/np.sqrt(2.0))

        ## Step 1  (Diffusion)
        for k in range (2,res_lon-2):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)

            if (k==2):
                B[k] = 1.0 + gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2) 
            elif(k==res_lon-3):
                B[k] = 1.0 + gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2) 
            else: 
                B[k] = 1.0 + 2*gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2) 

            if(k==res_lon-3):
                C[k] = 0.0
            else:
                C[k] = -gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)

            D[k] = mag[lat,k]  

        mag_new[lat,:] = tridiag2(res_lon,A,B,C,D)
        # Applying boundary conditions
        mag_new[lat,0] = mag_new[lat,res_lon-4]
        mag_new[lat,1] = mag_new[lat,res_lon-3]
        mag_new[lat,res_lon-2] = mag_new[lat,2]
        mag_new[lat,res_lon-1] = mag_new[lat,3]

        # Coefficient arrays for tridiagonal solver for step 2
        A = np.zeros(res_lon)  
        B = np.zeros(res_lon)
        C = np.zeros(res_lon)
        D = np.zeros(res_lon)

        ## Step 2  (Diffusion and Advection)
        for k in range (2,res_lon-2):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)

            if (k==2):
                B[k] = 1.0 + gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)
            elif(k==res_lon-3):
                B[k] = 1.0 + gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)
            else:
                B[k] = 1.0 + 2*gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)

            if(k==res_lon-3):
                C[k] = 0.0
            else:
                C[k] = -gamma*diff_lon(eta,lat_cen[lat])*dt/(r**2*lon_width[k]**2)

            D[k] = mag[lat, k]  

            D[k] = D[k] + dt*((1-2*gamma)*diffusion(mag_new[lat], d_lon, k, lon = 1, q = lat))

            D[k] = D[k] + dt*(advection(mag_new[lat], real_centre_array_lon, real_left_array_lon[k],
                                                           real_right_array_lon[k], dt, d_lon, k,t, lon = 1))
        mag_new1[lat,:] = tridiag2(res_lon,A,B,C,D)

        # Applying boundary conditions
        mag_new1[lat,0] = mag_new1[lat,res_lon-4]
        mag_new1[lat,1] = mag_new1[lat,res_lon-3]
        mag_new1[lat,res_lon-2] = mag_new1[lat,2]
        mag_new1[lat,res_lon-1] = mag_new1[lat,3]

        ## Step 3  (Diffusion and Advection)
        for k in range(2, res_lon-2):

            mag_new2[lat,k] = mag[lat,k] 
            mag_new2[lat,k] = mag_new2[lat,k] + dt*(diffusion(mag_new[lat], d_lon, k, lon = 1, q = lat)+ 
                                                    diffusion(mag_new1[lat], d_lon, k, lon = 1, q = lat))/2.0

            mag_new2[lat,k] = mag_new2[lat,k] + dt*(
                advection(mag_new[lat], real_centre_array_lon, real_left_array_lon[k], real_right_array_lon[k],
                          dt, d_lon, k, time, lon = 1)
            + advection(mag_new1[lat], real_centre_array_lon, real_left_array_lon[k], real_right_array_lon[k],
                          dt, d_lon, k, time, lon = 1))/2.0
        # Applying boundary conditions
        mag_new2[lat,0] = mag_new2[lat,res_lon-4]
        mag_new2[lat,1] = mag_new2[lat,res_lon-3]
        mag_new2[lat,res_lon-2] = mag_new2[lat,2]
        mag_new2[lat,res_lon-1] = mag_new2[lat,3]
    return mag_new2

"""
        Function used to update in latitude using RK-IMEX scheme, returns the updated value
        The function updates for each n_sub
        The function updates all latitude together
"""
@jit(nopython= True)
def rkimex_lat(mag,time,dt):
    for lon in range(res_lon):
        t = time
         # Coefficient arrays for tridiagonal solver step 1
        A = np.zeros(res_lat) 
        B = np.zeros(res_lat)
        C = np.zeros(res_lat)
        D = np.zeros(res_lat)

        mag_new = np.zeros((res_lat,res_lon))
        mag_new1 = np.zeros((res_lat,res_lon))
        mag_new2 = np.zeros((res_lat,res_lon))

        gamma = (1.0-1.0/np.sqrt(2.0))

        # Step 1 (Diffusion)
        for k in range (2,res_lat-2):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -gamma*diff_lat(eta,lon_cen[lon])*dt/(r**2*lat_width[k]**2)

            if (k==2):
                B[k] = 1.0 + gamma*diff_lat(eta,lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt
            elif(k==res_lon-3):
                B[k] = 1.0 + gamma*diff_lat(eta,lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt
            else: 
                B[k] = 1.0 + 2*gamma*diff_lat(eta,lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt

            if(k==res_lon-3):
                C[k] = 0.0
            else:
                C[k] = -gamma*diff_lat(eta,lon_cen[lon])*dt/(r**2*lat_width[k]**2)

            D[k] = mag[k,lon]  


        mag_new[:, lon] = tridiag2(res_lat,A,B,C,D)
        # Applying boundary conditions
        mag_new[0, lon] = mag_new[3, lon]
        mag_new[1, lon] = mag_new[2, lon]
        mag_new[res_lat-2, lon] = mag_new[res_lat-3, lon]
        mag_new[res_lat-1, lon] = mag_new[res_lat-4, lon]
        
        # Coefficient arrays for tridiagonal solver for step 2
        A = np.zeros(res_lat)  
        B = np.zeros(res_lat)
        C = np.zeros(res_lat)
        D = np.zeros(res_lat)

        ## Step 2 (Advection and Diffusion)
        for k in range (2,res_lat-2):
            if (k==2):
                A[k] = 0.0
            else:
                A[k] = -gamma*diff_lat(eta, lon_cen[lon])*dt/(r**2*lat_width[k]**2)

            if (k==2):
                B[k] = 1.0 + gamma*diff_lat(eta, lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt
            elif(k==res_lat-3):
                B[k] = 1.0 + gamma*diff_lat(eta, lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt
            else:
                B[k] = 1.0 + 2*gamma*diff_lat(eta, lon_cen[lon])*dt/(r**2*lat_width[k]**2) + gamma*esc(lat_cen[k],r,eta,tau)*dt

            if(k==res_lat-3):
                C[k] = 0.0
            else:
                C[k] = -gamma*diff_lat(eta, lon_cen[lon])*dt/(r**2*lat_width[k]**2)

            D[k] = mag[k, lon]  

            D[k] = D[k] + dt*((1-2*gamma)*diffusion(mag_new[:,lon], d_lat, k, lat = 1, q = lon))

            D[k] = D[k] + dt*(advection(mag_new[:,lon], real_centre_array_lat, real_left_array_lat[k],
                                                           real_right_array_lat[k], dt, d_lat, k,t, lat = 1))
            D[k] = D[k] - dt*((1-2*gamma)*mag_new[k, lon]*esc(real_centre_array_lat[k], r, eta, tau))
       
        mag_new1[:, lon] = tridiag2(res_lat,A,B,C,D)

        # Applying boundary conditions
        mag_new1[0, lon] = mag_new1[3, lon]
        mag_new1[1, lon] = mag_new1[2, lon]
        mag_new1[res_lat-2, lon] = mag_new1[res_lat-3, lon]
        mag_new1[res_lat-1, lon] = mag_new1[res_lat-4, lon]

        # Step 3 (Advection and Diffusion)
        for k in range(2, res_lat-2):

            mag_new2[k, lon] = mag[k, lon] 
            mag_new2[k, lon] = mag_new2[k, lon] + dt*(diffusion(mag_new[:,lon], d_lat, k, lat = 1, q = lon)
                                                      + diffusion(mag_new1[:,lon], d_lat, k, lat = 1, q = lon))/2.0

            mag_new2[k, lon] = mag_new2[k, lon] + dt*(
                advection(mag_new[:, lon], real_centre_array_lat, real_left_array_lat[k], real_right_array_lat[k],
                          dt, d_lat, k, time, lat = 1)
            + advection(mag_new1[:, lon], real_centre_array_lat, real_left_array_lat[k], real_right_array_lat[k],
                          dt, d_lat, k, time, lat = 1))/2.0
            mag_new2[k, lon] = mag_new2[k, lon] - dt*((mag_new[k, lon]*esc(lat_cen[k], r, eta, tau)) +
                                      (mag_new1[k, lon]*esc(lat_cen[k], r, eta, tau)))/2.0
        # Applying boundary conditions
        mag_new2[0, lon] = mag_new2[3, lon]
        mag_new2[1, lon] = mag_new2[2, lon]
        mag_new2[res_lat-2, lon] = mag_new2[res_lat-3, lon]
        mag_new2[res_lat-1, lon] = mag_new2[res_lat-4, lon]
    return mag_new2

# # Solving

# update using Explicit scheme
time = np.zeros(num_files+1)
time[0] = 0.0

for i in tqdm(range(num_files)):
    time[i+1] = time[i] + dt_glb  # Calculate current time step
    for j in range(nsub):
        if i%2==1:
            mag_1 = lat_update(mag, nsub, dt/2.0, dt/2.0)
            mag_2 = lon_update(mag_1, nsub, dt, dt)
            mag = lat_update(mag_2, nsub, dt/2.0, dt/2.0)
            
        else:
            mag_1 = lon_update(mag, nsub, dt/2.0, dt/2.0)
            mag_2 = lat_update(mag_1, nsub, dt, dt)
            mag = lon_update(mag_2, nsub, dt/2.0, dt/2.0)
        
    B_Explicit[i+1] = mag
    
time = np.zeros(num_files+1)
time[0] = 0.0

# update using RK-IMEX scheme   
for i in tqdm(range(num_files)):
    time[i+1] = time[i] + dt_glb  # Calculate current time step
    for j in range(nsub):
        if i%2==1:
            mag_1 = rkimex_lat(mag, nsub, dt/2.0, dt/2.0)
            mag_2 = rkimex_lon(mag_1, nsub, dt, dt)
            mag = rkimex_lat(mag_2, nsub, dt/2.0, dt/2.0)
            
        else:
            mag_1 = rkimex_lon(mag, nsub, dt/2.0, dt/2.0)
            mag_2 = rkimex_lat(mag_1, nsub, dt, dt)
            mag = rkimex_lon(mag_2, nsub, dt/2.0, dt/2.0)
        
    B_RK_IMEX[i+1] = mag

## To save the files
# variable_name = f"numeric_2D_num_{res}.npy"
# np.save(variable_name, B_explicit)

# variable_name = f"numeric_2D_rk_{res}.npy"
# np.save(variable_name, B_RK_IMEX)

# ## Plotting

plt.imshow(B_Explicit[-1], origin="lower")#, aspect = 0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Initial Magnetic Field Distribution')
plt.grid()
plt.colorbar()
plt.show()

