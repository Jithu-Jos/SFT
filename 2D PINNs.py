#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Importing required modules

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
import time as tt
from tqdm import tqdm
##End importing required modules


# ## Code Units

# In[ ]:


##The time in years for which the simulation is conducted
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
pi = tf.constant(np.pi)


# ## Constants and Parameters

# In[ ]:


##solar radius
r = Rsun / L_unit #(6.95e-10 cm)

## Diffusion constant
eta = 500e+10    #500km2/s      
eta = eta/(L_unit * V_unit) 

## Advection Velocity
u_0 = -12.5e+2     # Advection velocity in m/s
u_0 = u_0 / V_unit  
lam0 = 75.0 * np.pi / 180.0  
lam00 = tf.constant(lam0)

## Decay Time
tau = 5.0 * 365.25 * 24.0 * 3600.0     # Decay time in seconds (5.0 years)
tau = tau / T_unit

## Initial amplitude of magnetic field
B0 = 10.0    # in Gauss
B0 = B0 / B_unit


# In[ ]:


# initial condition
def initial(x):
    theta = pi/2 - x[:,0:1]*np.pi
    phi = x[:,1:2]*2*np.pi
    
    Bmax = 100.0/B_unit  # Maximum magnetic field strength
    b = 0.4
    
    tilt = 15.0*deg_2_rad
    theta0 = np.pi/2.0
    xphi =  2.0*deg_2_rad/(np.sin(theta0)*np.tan(tilt))
    
    np_phi_minus = 1.5*np.pi + xphi
    np_phi_plus = 1.5*np.pi - xphi
    np_theta_plus = (np.pi / 2) - 1.0*deg_2_rad - 15.0*deg_2_rad
    np_theta_minus = (np.pi / 2) + 1.0*deg_2_rad - 15.0*deg_2_rad

    theta_minus = tf.constant(np_theta_minus, dtype=tf.float32)
    theta_plus = tf.constant(np_theta_plus, dtype=tf.float32)
    phi_minus = tf.constant(np_phi_minus, dtype=tf.float32)
    phi_plus = tf.constant(np_phi_plus, dtype=tf.float32)

    cos_theta_plus = tf.cos(theta_plus)
    cos_theta_minus = tf.cos(theta_minus)
    sin_theta_plus = tf.sin(theta_plus)
    sin_theta_minus = tf.sin(theta_minus)

    cos_beta_plus = (cos_theta_plus * tf.cos(theta) +
                     sin_theta_plus * tf.sin(theta) * tf.cos(phi - phi_plus))

    cos_beta_minus = (cos_theta_minus * tf.cos(theta) +
                      sin_theta_minus * tf.sin(theta) * tf.cos(phi - phi_minus))

    cos_rho_0 = (cos_theta_plus * cos_theta_minus +
                 sin_theta_plus * sin_theta_minus * tf.cos(phi_plus - phi_minus))

    delta = b * tf.acos(cos_rho_0)

    B_plus = Bmax * tf.exp(-2 * (1 - cos_beta_plus) / delta**2)
    B_minus = Bmax * tf.exp(-2 * (1 - cos_beta_minus) / delta**2)

    delta_Br = B_plus - B_minus
    return delta_Br


# ## SFT equation

# In[ ]:


# Advection velocity
def adv(lam):
    u = u_0 * tf.sin(np.pi * lam / lam00)
    return u

# Differential rotation
def diff_rot(lam):
    th = pi/2 - lam
    omega = 0.18 - 2.36 * (tf.cos(th)**2) - 1.787 * (tf.cos(th)**4)    #deg/day    Snodgrass and Ulrich (1990)
    omega = omega*np.pi/180.0
    return omega*T_unit/(24.0*60.0*60.0)
     
# 2D SFT equation
def pde_SFT_2D(x, y): # x[0] = latitude, x[1] = longitude, x[2] = time, y = Magnetic field
    
    # Compute the derivatives of B
    dB_t = dde.grad.jacobian(y, x, i=0, j=2)
    dB1_lam = dde.grad.jacobian(y*(adv(np.pi*x[:,0:1]) - eta*tf.tan(np.pi*x[:,0:1])),x,j=0)/np.pi
    dB2_lam = dde.grad.hessian(y,x,i=0,j=0)/(np.pi**2)
    dB1_phi = dde.grad.jacobian(y,x,j=1)/(2*np.pi)
    dB2_phi = dde.grad.hessian(y,x,i=1,j=1)/((2*np.pi)**2)
    
    # return the PDE equation for the SFT model
    return (
             dB_t 
           - dB1_lam 
           - eta * dB2_lam 
        + diff_rot(x[:,0:1]*np.pi)*dB1_phi
        - eta*dB2_phi/(tf.cos(x[:,0:1]*np.pi)**2)
           + y*(
               1/tau 
        + adv(np.pi*x[:,0:1])*tf.tan(x[:,0:1]*np.pi) 
               - eta * ((1/tf.cos(x[:,0:1]*np.pi))**2)
           )
          )

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


## To define the boundary
def boundary_lam(x, on_boundary):
    return (on_boundary and (isclose(x[0], lam_max) or isclose(x[0], lam_min)))


def boundary_phi(x, on_boundary):
    return (on_boundary and (isclose(x[1], phi_max) or isclose(x[1], phi_min)))
    


# ## PINNs

# In[ ]:


## Simulation box
lam_min = -0.40  # -0.40 * pi in radian
lam_max = 0.40  # 0.40 * pi in radian

phi_min = 0.0   # 0 in radian
phi_max = 1.0   # 1.0 * (2*pi) in radian

Tmin = 0.0
Tmax = 1.0

geom = dde.geometry.geometry_2d.Rectangle( xmin = [lam_min ,phi_min ], xmax = [lam_max ,phi_max] )      # Latitude interval
timedomain = dde.geometry.TimeDomain(Tmin, Tmax)   # Time interval
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

layer_size = [3] + [32] * 8 + [1]   # Define the layer sizes for the neural network
activation = "tanh"  
initializer = "Glorot uniform"  
net = dde.nn.FNN(layer_size, activation, initializer)

# print("eta",eta)
# print("u_0",u_0)
# print("tau",tau)


# In[ ]:


bc_lam = dde.icbc.NeumannBC(geomtime, lambda x:0 , boundary_lam)   # Boundary Condition for latitude
bc_phi = dde.icbc.PeriodicBC(geomtime, 1, boundary_phi)   # Boundary Condition for longitude 
ic = dde.icbc.IC(geomtime, initial, lambda _, on_initial: on_initial)


data = dde.data.TimePDE(
    geomtime,
    pde_SFT_2D,
    [
     bc_lam, bc_phi, ic
       ],
    num_domain =   173567,   
    num_boundary = 20022,     
    num_initial =  28838,     
    num_test = 1000
)


# In[ ]:


# Create the model using the defined data and neural network
model01 = dde.Model(data, net)

model01.compile("adam", lr=5.2235258019522066e-04         # Optimizer used and the learning rate 
             ,loss_weights=[2391, 1787, 4967, 1892]        # Weight on individual loss 
              )
losshistory, train_state = model01.train(iterations=199495,     # Total iterations
                                         display_every=5000)    # Display the loss in every n iterations

dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# In[ ]:


model01.compile("L-BFGS",loss_weights=[1, 1,1, 1])    # Compile the model with the L-BFGS optimizer and loss weights          # Optimizer used and the learning rate
losshistory, train_state = model01.train(display_every = 5000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# ## To produce pinn solutions for different resolutions

# In[ ]:


lat_max = 0.40*180.0  
lat_min = -0.40*180.0 

lon_max = 360.0 
lon_min = 0.0  

num_files = 1000
dt = (Tmax - Tmin) / num_files
time_array = np.linspace(Tmin,Tmax,num_files+1)

res_lat_values = [36, 72, 144, 288]
res_lon_values = [90, 180, 360, 720]

latitudes = {}
longitudes = {}

for res_lat in res_lat_values:
    dlat = (lat_max - lat_min) / res_lat
    lat_left = np.zeros(res_lat)
    lat_right = np.zeros(res_lat)
    lat_cen = np.zeros(res_lat)
    
    for i in range(len(lat_cen)):
        lat_left[i] = (lat_min + i * dlat) * deg_2_rad
        lat_right[i] = (lat_min + (i + 1) * dlat) * deg_2_rad
        lat_cen[i] = (lat_right[i] + lat_left[i]) / 2.0
    
    latitudes[res_lat] = {
        'lat_cen': lat_cen
    }

for res_lon in res_lon_values:
    dlon = (lon_max - lon_min) / res_lon  
    lon_left = np.zeros(res_lon) 
    lon_right = np.zeros(res_lon)  
    lon_cen = np.zeros(res_lon)  
    
    for i in range(len(lon_cen)):
        lon_left[i] = (lon_min + i * dlon) * deg_2_rad  
        lon_right[i] = (lon_min + (i + 1) * dlon) * deg_2_rad  
        lon_cen[i] = (lon_right[i] + lon_left[i]) / 2.0  
        
    longitudes[res_lon] = {
        'lon_cen': lon_cen
    }
    
# Producing the results
for i in range(len(res_lat_values)):
    k = res_lat_values[i]
    kk = res_lon_values[i]
    print(k,kk)
    x = latitudes[k]['lat_cen']/np.pi
    yy = longitudes[kk]['lon_cen']/(2*np.pi)
    X, Y, T, re = [], [], [], []
    for j in tqdm(x):
        for k in yy:
            lat = np.ones(len(time_array)) * j
            phi = np.ones(len(time_array)) * k
            con = np.stack((lat, phi, time_array), axis=1)
            y = model01.predict(con) 
            re.append(y)       
    re = np.reshape(re, (len(x), len(yy), len(time_array)))
    re = np.transpose(re,(2,0,1))
    variable_name = f"pinns_2D_{res_lon_values[i]}.npy"
    np.save(variable_name,re)


# ## Plotting

# In[ ]:


plt.figure(figsize=(10, 6))
im = plt.imshow(re[-1], origin = "lower", extent=[lon_min, lon_max, lat_min, lat_max])
plt.xlabel("phi")
plt.ylabel("latitude")
plt.title("PINNs")
plt.colorbar(im,  location="bottom", aspect=70, extend="both", label="B")
plt.title("Magnetic field solving the 2D SFT equation")
plt.show()

