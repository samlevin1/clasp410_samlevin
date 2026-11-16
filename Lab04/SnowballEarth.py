import numpy as np
import matplotlib.pyplot as plt
from snowball_functions import temp_warm, insolation

#Constants
S0=1370
C=4.2e6
rho=1020
dz=50
R_earth=6378e3
sigma=5.67e-8
albedo_gnd=0.3
albedo_ice=0.6

#Grid
npoints=18
dlat= 180/npoints
lats= np.linspace(dlat/2., 180-dlat/2., npoints)
dy= R_earth*np.radians(dlat) #meters
dt=365*24*3600 #one year converted to seconds
nsteps=10000

#Build K (the second derivative) with no flux at the poles
K= np.zeros((npoints, npoints))
for i in range(npoints):
    K[i, i]= -2 
    if i>0:
        K[i, i-1] = 1
    if i< npoints-1: 
        K[i, i+1] = 1 
K[0, 1 ]=2
K[npoints-1, npoints-2] =2
K= K/dy**2 

#Now, build B (first derivative) with zero at the poles
B= np.zeros((npoints, npoints))
for i in range (1, npoints-1):
    B[i, i+1]= 1
    B[i, i-1]= -1
B= B/(2*dy)
I= np.eye(npoints)

#Spherical correction pieces
Axz= 2* np.pi* R_earth* np.sin(np.radians(lats))* dz
dAxz= B@Axz

#Insolation - it doesn't change so only need to compute once
insol= insolation(S0, lats)
T_target= temp_warm(lats)

#Solver with a constant albedo
def run_model(lam, emiss, T0):
    Linv= np.linalg.inv(I-lam* dt* K)
    T= T0.copy()
    for step in range(nsteps):
        T_K= T+273.15
        corr=lam* dt* (B@T)* dAxz/Axz
        forcing= dt/(rho* C* dz) * (insol*(1-albedo_gnd)-emiss* sigma* T_K**4)
        T= Linv @(T+corr+forcing)
    return T

#Solver now with a dynamic albedo and solar multiplier
def run_dynamic(T0, gamma, lam, emiss, flash=False):
    Linv= np.linalg.inv(I-lam* dt* K)
    T= T0.copy()
    albedo= np.zeros(npoints)
    if flash:
        albedo[:]= albedo_ice #This is the flash freeze. Starting with ice albedo everywhere
    else:
        loc_ice= T <= -10
        albedo[loc_ice]= albedo_ice
        albedo[~loc_ice]= albedo_gnd
    for step in range(nsteps):
        T_K =T+273.15
        corr=lam* dt* (B@T)* dAxz/Axz
        forcing= dt/(rho* C* dz) * (gamma* insol*(1-albedo)-emiss* sigma* T_K**4)
        T= Linv @(T+corr+forcing)
#Update the albedo for the next step
        loc_ice = T <= -10
        albedo[loc_ice] = albedo_ice
        albedo[~loc_ice] = albedo_gnd
    return T

#----PART 2------

#Figure 1
#Emissivity fixed at 1 and lamda varied
plt.figure(1)
for lam_test in [0,25,50,100,150]:
    T_eq= run_model(lam_test, 1.0, T_target)
    plt.plot(lats-90, T_eq, label='lamda = '+str(lam_test))
plt.plot(lats-90, T_target, 'k--', label='temp_warm')
plt.title('Figure 1: Effect of Diffusivity (lambda)')
plt.xlabel('Latitude')
plt.ylabel('Temperature (C)')
plt.legend()

#Figure 2
# Emissivity varied but lamda fixed at 100
plt.figure(2)
for e_test in [0.4,0.5,0.6,0.7,0.8,1.0]:
    T_eq= run_model(100, e_test, T_target)
    plt.plot(lats-90, T_eq, label='emissivity = '+str(e_test))
plt.plot(lats-90, T_target, 'k--', label='temp_warm')
plt.title('Figure 2: Effect of Emisivity')
plt.xlabel('Latitude')
plt.ylabel('Temperature (C)')
plt.legend()

#Search the grid for the best lambda and emissivity
lam_vals= np.arange(10,101,10)
emiss_vals= np.arange(0.60,0.81,0.02)
rms=np.zeros((lam_vals.size, emiss_vals.size))

for i in range(lam_vals.size):
    for j in range(emiss_vals.size):
        T_eq= run_model(lam_vals[i], emiss_vals[j], T_target)
        rms[i,j]= np.sqrt(np.mean((T_eq-T_target)**2))

i_best, j_best= np.unravel_index(np.argmin(rms), rms.shape)
print('best lamda =', lam_vals[i_best], 'emissivity =', round(emiss_vals[j_best], 2),
      'RMS =', round(rms[i_best, j_best], 2))

#Using the best values from here on out
lam_best=30
emiss_best=0.72

#Figure 3: Best fit vs. temp_warm
plt.figure(3)
plt.plot(lats-90, T_target, 'k--', label='temp_warm')
plt.plot(lats-90, run_model(lam_best, emiss_best, T_target), label='best fit')
plt.title('Figure 3: Best fit model vs warm Earth')
plt.xlabel('Latitude')
plt.ylabel('Temperature (C)')
plt.legend()

#----PART 3-----
T_hot= np.zeros(npoints)+60
T_cold= np.zeros(npoints)-60

T_hot_eq= run_dynamic(T_hot, 1.0, lam_best, emiss_best)
T_cold_eq= run_dynamic(T_cold, 1.0, lam_best, emiss_best)
T_flash_eq= run_dynamic(T_target, 1.0, lam_best, emiss_best, flash=True)

#Figure 4. The three initial conditions
plt.figure(4)
plt.plot(lats-90, T_hot_eq, label='Hot Start')
plt.plot(lats-90, T_cold_eq, label='Cold Start')
plt.plot(lats-90, T_flash_eq, label='Flash Freeze')
plt.title('Figure 4: Equilibria for different initial conditions')
plt.xlabel('Latitude')
plt.ylabel('Temperature (C)')
plt.legend()

#---PART 4----
gammas_up= np.round(np.arange(0.4, 1.401, 0.05 ), 2)
gammas_down= gammas_up[::-1][1:]
gammas= np.concatenate([gammas_up, gammas_down])

weights= np.sin(np.radians(lats)) #Area weights for the global average

T=T_cold.copy()
means=[]
for g in gammas:
    T= run_dynamic(T,g,lam_best,emiss_best) #Makes the previous result the next start
    means.append(np.sum(T*weights)/np.sum(weights))

n_up= gammas_up.size

#Figure 5: Hysteresis loop
plt.figure(5)
plt.plot(gammas[:n_up], means[:n_up], 'o-', label='Increasing gamma')
plt.plot(gammas[n_up:], means[n_up:], 's--', label='Decreasing gamma')
plt.title('Figure 5: Global Mean Temperature vs Solar Multiplier')
plt.xlabel('Solar multiplier (gamma)')
plt.ylabel('Global mean temperature (C)')
plt.legend()

plt.show()