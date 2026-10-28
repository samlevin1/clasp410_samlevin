#Lab03 Energy Balance Climate Models

#Layer 0 is the ground, layers 1 to N are the atmosphere.
#x0=sigma * T0^4 for the ground, xi=e * sigma * Ti^4 for each layer.
#S is the sunlight absorbed by the planet: S=(1 - albedo) * S0 / 4

#Ground equation (row 0):
#  x0-(sum of (1-e)^(j-1) * xj) = S

# Layer i equation (rows 1 to N):
#  2*xi-e*(1-e)^(i-1)*x0-(sum of e*(1-e)^(|i-j|-1)*xj for j not i) = 0

#Written as Ax = b, A holds the coefficients, x holds the fluxes, and b is all zeros except for S.
#Nuclear winter: the top layer absorbs the sunlight, so S moves into b[N].

import numpy as np                
import matplotlib.pyplot as plt

sigma = 5.6703e-8                 #FROM HANDOUT (Stefan-Boltzmann constant)

def nlayer(nlayers, S0, emiss, albedo, nuclear=False, debug=False):  #debug keyword

    #sunlight absorbed
    S=(1 - albedo) * S0 / 4

    #make empty A and b         
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    #Fill in A                   #loop structure
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==0:
                #ground row
                if j==0:
                    A[i, j]=1
                else:
                    A[i, j]= -(1 - emiss)**(j - 1)
            else:
                #atmosphere rows
                if i==j:
                    A[i, j]=2
                elif j==0:
                    A[i, j]= -emiss * (1 - emiss)**(i - 1)
                else:
                    A[i, j]= -emiss * (1 - emiss)**(abs(i - j) - 1)

            if debug:             
                print(f'A[i={i},j={j}] = {A[i, j]}')

    #Fill in b
    if nuclear:
        b[nlayers]= S
    else:
        b[0]= S

    #Solve for the fluxes        (Use solve not inverse)
    fluxes = np.linalg.solve(A, b)

    #convert fluxes to temperatures with Stefan-Boltzmann
    temps=np.zeros(nlayers+1)
    temps[0]= (fluxes[0] / sigma)**0.25
    for i in range(1, nlayers+1):
        temps[i]= (fluxes[i] / (emiss * sigma))**0.25

    if debug:
        print('A =', A)
        print('b =', b)
        print('fluxes =', fluxes)
        print('temps =', temps)

    return temps

#Quick check against class example (1 layer, emissivity 1)
print(nlayer(1, 1350, 1.0, 0.33))

albedo_earth = 0.3

#Experiment 1: one layer, change emissivity
emis_list= np.linspace(0.05, 1.0, 96)
surface_temps=[]

for e in emis_list:
    t=nlayer(1, 1350, e, albedo_earth)
    surface_temps.append(t[0])

#Find the first emissivity that gets us to 288 K
e_needed=0
for k in range(len(emis_list)):
    if surface_temps[k] >= 288:
        e_needed=emis_list[k]
        break
print('Emissivity needed for 288 K:', e_needed)

plt.figure()
plt.plot(emis_list, surface_temps)
plt.xlabel('Emissivity')
plt.ylabel('Surface temperature (K)')
plt.title('Surface temp vs emissivity (1 layer)')

#Experiment 2: emissivity 0.255, change number of layers
layer_list=[]
layer_temps=[]
n_needed=0

for n in range(1, 31):
    t=nlayer(n, 1350, 0.255, albedo_earth)
    layer_list.append(n)
    layer_temps.append(t[0])
    if t[0] >= 288 and n_needed == 0:
        n_needed= n
print('Layers needed for about 288 K:', n_needed)

plt.figure()
plt.plot(layer_list, layer_temps)
plt.xlabel('Number of layers')
plt.ylabel('Surface temperature (K)')
plt.title('Surface temp vs layers (emissivity 0.255)')

#Altitude profile using the layer count we found
profile = nlayer(n_needed, 1350, 0.255, albedo_earth)
altitude = list(range(n_needed + 1))   # layer number stands in for altitude

plt.figure()
plt.plot(profile, altitude)
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (layer number)')
plt.title('Earth altitude profile')
plt.show()

albedo_venus = 0.75

venus_layers=[]
venus_temps=[]
n_venus=0

for n in range(2, 201):
    t=nlayer(n, 2600, 1.0, albedo_venus)
    venus_layers.append(n)
    venus_temps.append(t[0])
    if t[0] >= 700 and n_venus== 0:
        n_venus= n
print('Layers needed for Venus:', n_venus)

plt.figure()
plt.plot(venus_layers, venus_temps)
plt.xlabel('Number of layers')
plt.ylabel('Surface temperature (K)')
plt.title('Venus surface temp vs layers')
plt.show()

nw=nlayer(5, 1350, 0.5, albedo_earth, nuclear=True)
print('Nuclear winter surface temperature:', nw[0])

nw_altitude= [0, 1, 2, 3, 4, 5]

plt.figure()
plt.plot(nw, nw_altitude)
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (layer number)')
plt.title('Nuclear winter altitude profile')
plt.show()