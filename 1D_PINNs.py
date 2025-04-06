##Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf
import time as tt
from tqdm import tqdm
##End importing required modules

# ## Code Units

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
pi = tf.constant(np.pi)

# ## Constants and Parameters

##solar radius
r = Rsun / L_unit #(6.95e-10 cm)

## Diffusion constant
eta = 500e+10    # 500 cm2/s
eta = eta / (L_unit * V_unit)        

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

# Source term parameters
ahat = 0.00185  
bhat = 48.7  
chat = 0.71
sourcescale = 0.003 * 10000 / B_unit
cycleper = 11.0 * 365.25   # Solar cycle duration in days (11 years)

# Integration by trapezoidal rule
def trapezoidal_integration(func, a, b, n):
    x = np.linspace(a, b, n+1).astype(np.float32)
    y = func(x)
    y_left = y[:-1]
    y_right = y[1:]
    dx = (b - a) / n
    integrand = (y_left + y_right) * dx / 2
    integral = tf.reduce_sum(integrand)
    return integral

# ## SFT equation

def source(latitude, t):
    latitude = tf.cast(latitude, tf.float32)
    t = tf.cast(t, tf.float32)
    tc = 12.0 * ((((t / cycleper) % 1) * cycleper / 365.25))
    ampli = sourcescale * ahat * tc**3 / (tf.exp(tc**2 / bhat**2) - chat)
    cycleno = tf.cast(t // cycleper, tf.float32) + 1
    evenodd = 1 - 2 * (cycleno % 2)
    lambda0 = 26.4 - 34.2 * ((t / cycleper) % 1) + 16.1 * ((t / cycleper) % 1)**2
    fwhm = 6.0
    joynorm = 0.5 / tf.sin(20.0 / 180 * np.pi)
    
    bandn1 = evenodd * ampli * tf.exp(-(latitude - lambda0 - joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    bandn2a = -evenodd * ampli * tf.exp(-(latitude - lambda0 + joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    bands2a = evenodd * ampli * tf.exp(-(latitude + lambda0 - joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    bands1 = -evenodd * ampli * tf.exp(-(latitude + lambda0 + joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    
    Nfine = 180 + 1
    thetaf = tf.linspace(0.0, tf.constant(np.pi, dtype=tf.float32), Nfine)
    latitudef = 90.0 - thetaf * 180 / np.pi
    
    bandn1f = evenodd * ampli * tf.exp(-(latitudef - lambda0 - joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    bandn2af = -evenodd * ampli * tf.exp(-(latitudef - lambda0 + joynorm * tf.sin(lambda0 / 180 * np.pi))**2 / (2 * fwhm**2))
    
    fluxband1 = trapezoidal_integration(lambda x: -tf.sin(x) * bandn1f, 0, np.pi, Nfine - 1)
    fluxband2=trapezoidal_integration(lambda x: -tf.sin(x) * bandn2af, 0, np.pi,Nfine-1)

    fluxcorrection=1.0
    if (ampli != 0): fluxcorrection=-fluxband1/fluxband2 

    bandn2=fluxcorrection*bandn2a
    bands2=fluxcorrection*bands2a

    value=bandn1+bandn2+bands1+bands2

    return value

# Advection velocity
def adv(lam):
    u = u_0 * tf.sin(np.pi * lam / lam00)
    return u

# 1D SFT equation
def pde_SFT(x, y):  # x[0] = latitude, x[1] = time, y = Magnetic field
    # Compute the derivatives of B
    dB_t = dde.grad.jacobian(y, x, j=1)
    dB1_lam = dde.grad.jacobian(y * (adv(np.pi * x[:, 0:1]) - eta * tf.tan(np.pi * x[:, 0:1])), x, j=0) / np.pi
    dB2_lam = dde.grad.hessian(y, x, i=0, j=0) / (np.pi ** 2)
    
    # Calculate the source term at given latitude and time
    sou = source(x[:, 0:1] * 180, x[:, 1:2] * simul_time * 365.25)

    # return the PDE equation for the SFT model
    return (dB_t - dB1_lam - eta * dB2_lam
            + y * (1 / tau + adv(np.pi * x[:, 0:1]) * tf.tan(x[:, 0:1] * np.pi) - eta * (1 / tf.cos(x[:, 0:1] * np.pi)) ** 2)
            - sou)

# Set the initial condition for the magnetic field
def init(x):
    B = B0 * tf.sin(x[:, 0:1] * np.pi)
    return B

# Define the boundary for the computational domain
def boundary(x, on_boundary):
    return on_boundary

# ## PINNs

lam_max = 0.40  # Maximum latitude in radians (0.40*pi)
lam_min = -0.40 # Minimum latitude in radians (-0.40*pi)

Tmax = 1.0
Tmin = 0.0

geom = dde.geometry.Interval(lam_min, lam_max)      
timedomain = dde.geometry.TimeDomain(Tmin, Tmax)   
geomtime = dde.geometry.GeometryXTime(geom, timedomain)   

layer_size = [2] + [41] * 10 + [1]   # Define the layer sizes for the neural network
activation = "tanh"
initializer = "Glorot uniform"    
net = dde.nn.FNN(layer_size, activation, initializer)   

# print("eta", eta)  
# print("u_0", u_0)   
# print("tau", tau)   

ic = dde.icbc.IC(geomtime, init, lambda _, on_initial: on_initial)    # Define the initial condition
bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary, component=0)   # Define the Neumann boundary condition

data = dde.data.TimePDE(
    geomtime,
    pde_SFT,
    [bc, ic],
    num_test=1000,
    num_domain = 87460,
    num_boundary = 2356,
    num_initial = 2787
)

# Create the model using the defined data and neural network
model01 = dde.Model(data, net)   
# checkpointer = dde.callbacks.ModelCheckpoint("./model/model01", verbose=1, save_better_only=True, period=1000)
model01.compile("adam"
                , lr = 0.002203
                , loss_weights=[3233, 48, 4975] 
               )   
losshistory, train_state = model01.train(iterations=35000, display_every=1000
#                                          , callbacks=[checkpointer]
                                        )  
# dde.saveplot(losshistory, train_state, issave=True, isplot=False)

model01.compile("L-BFGS"
                , loss_weights=[20, 1, 10]
               )   # Compile the model with the L-BFGS optimizer and loss weights
losshistory, train_state = model01.train(display_every=500, model_save_path= "./model/model01")   
dde.saveplot(losshistory, train_state, issave=True, isplot=True) 

# model01.compile("L-BFGS")   # Compile the model with the L-BFGS optimizer
# losshistory, train_state = model01.train(display_every=1000, model_save_path= "./model/model01")   # Train the model 
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)   # Save and plot the loss history


# ## To produce pinn solutions for different resolutions

lat_max = 70.0  
lat_min = -70.0 

num_files = 1000
dt = (Tmax - Tmin) / num_files
time_array = np.linspace(Tmin,Tmax,num_files+1)

res_lat_values = np.array([128,256,384,512,1024])

latitudes = {}

for res_lat in res_lat_values:
    dlat = (lat_max - lat_min) / res_lat
    lat_left = np.zeros(res_lat)
    lat_right = np.zeros(res_lat)
    lat_cen = np.zeros(res_lat)
    lat_width = np.zeros(res_lat)
    
    for i in range(len(lat_cen)):
        lat_left[i] = (lat_min + i * dlat) * deg_2_rad
        lat_right[i] = (lat_min + (i + 1) * dlat) * deg_2_rad
        lat_cen[i] = (lat_right[i] + lat_left[i]) / 2.0
        lat_width[i] = lat_right[i] - lat_left[i]
    
    latitudes[res_lat] = {
        'lat_cen': lat_cen
    }

# Producing the results
for i in range(len(res_lat_values)):
    k = res_lat_values[i]
    print(k)
    x = latitudes[k]['lat_cen']/np.pi
    X, T, resu = [], [], []
    for j in x:
        lat = np.ones(len(time_array)) * j
        con = np.stack((lat, time_array), axis=1)
        X.append(np.ones(len(time_array)) * j)
        T.append(time_array)
        y = model01.predict(con)
        resu.append(y)

    X = np.array(X)
    T = np.array(T)
    resu = np.array(resu)
    X = X.reshape(len(x) * len(time_array))
    T = T.reshape(len(x) * len(time_array))
    resul = resu.reshape(len(x) * len(time_array))
    meth1 = np.copy(resu)
    meth1_new = meth1.reshape((len(x), len(time_array)))
    meth1_new1 = meth1_new.T
    variable_name = f"pinns_1D_{res_lat_values[i]}.npy"
    np.save(variable_name,meth1_new1)

# ## Plotting
im = plt.imshow(meth1_new1.T*10, vmin=-5.0, vmax=5.0, aspect="auto", origin="lower", extent=[0, simul_time, lat_min, lat_max])
plt.xlabel("Time")
plt.ylabel("Latitude")
plt.title("PINNs")
plt.colorbar(im, ax=axs, location="bottom", aspect=70, extend="both", label="B")
plt.title("Magnetic field solving the 1D SFT equation")
plt.show()
