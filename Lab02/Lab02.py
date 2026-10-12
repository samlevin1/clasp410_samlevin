#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 2 for CLaSP
410.
To reproduce the plots shown in the lab report, Run this file with python lab2.py and all figures will be saved in the same folder.
'''

import numpy as np                   #for arrays and math
import matplotlib.pyplot as plt      #for making plots
from scipy.integrate import ode      #this has the rk45 solver (dopri5)

def competition(t, N, a=1, b=2, c=1, d=3):
    """Competition equations (Eq. 1 and 2). N[0] is species 1, N[1] is species 2."""
    N1= N[0]
    N2= N[1]
    #logistic growth minus the harm from the other species
    dN1= a * N1 * (1 - N1) - b * N1 * N2
    dN2= c * N2 * (1 - N2) - d * N1 * N2
    return np.array([dN1, dN2])


def predprey(t, N, a=1, b=2, c=1, d=3):
    """Predator-prey equations (Eq. 4 and 5). N[0] is prey, N[1] is predator."""
    N1= N[0]
    N2= N[1]
    #prey grow on their own and get eaten
    dN1= a * N1 - b * N1 * N2
    #predators die on their own and grow by eating prey
    dN2= -c * N2 + d * N1 * N2
    return np.array([dN1, dN2])


def lorenz(t, X, sigma=10.0, r=23.0, b=8.0 / 3.0):
    """Lorenz equations. X[0] is x, X[1] is y, X[2] is z."""
    x= X[0]
    y= X[1]
    z= X[2]
    dx= -sigma * x + sigma * y
    dy= -x * z + r * x - y
    dz= x * y - b * z
    return np.array([dx, dy, dz])

def euler_solve(func, N0, dt, t_final, params):
    """Euler method. Returns the time array and an array of populations."""
    #number of steps (rounded so that we don't lose a step to decimals)
    nsteps= int(round(t_final / dt))

    #empty arrays to fill in
    time= np.zeros(nsteps + 1)
    N= np.zeros((nsteps + 1, len(N0)))

    #first row is the starting condition
    N[0, :] = N0

    #take one step at a time
    for i in range(nsteps):
        time[i + 1] = time[i] + dt
        slope = func(time[i], N[i, :], *params)
        N[i + 1, :] = N[i, :] + dt * slope

    return time, N


def rk45_solve(func, N0, dt, t_final, params):
    """rk45 method from scipy. Returns the same things as euler_solve."""
    #make the solver and pick dopri5 (this is rk45)
    solver= ode(func).set_integrator('dopri5', nsteps=10000)
    solver.set_initial_value(N0, 0)
    solver.set_f_params(*params)

    #lists to collect the answers
    time= [0.0]
    N= [np.array(N0)]

    #keep going until we reach the final time
    while solver.successful() and solver.t < t_final - 1e-9:
        solver.integrate(solver.t + dt)
        time.append(solver.t)
        N.append(solver.y)

    return np.array(time), np.array(N)

print("TESTS")

def decay(t, y, k=1.0):
    """Test equation dy/dt = -k*y."""
    return np.array([-k * y[0]])

y0= np.array([1.0])

# Test 1: compare both solvers to the exact answer at t = 5
t1, y1= euler_solve(decay, y0, 0.01, 5.0, (1.0,))
t2, y2= rk45_solve(decay, y0, 0.5, 5.0, (1.0,))
print("Test 1: Euler error", abs(y1[-1, 0] - np.exp(-5.0)))
print("Test 1: rk45 error ", abs(y2[-1, 0] - np.exp(-5.0)))

# Test 2: Euler is first order, so cutting dt in half should cut the error in half
for dt_test in [0.1, 0.05, 0.025]:
    tt, yy= euler_solve(decay, y0, dt_test, 5.0, (1.0,))
    print("Test 2: dt =", dt_test, "error =", abs(yy[-1, 0] - np.exp(-5.0)))

# Test 3: competition model should stay at the equilibrium from Eq. 3
a, b, c, d = 3, 2, 4, 3
N1_eq= c * (a - b) / (c * a - b * d)
N2_eq= a * (c - d) / (c * a - b * d)
tt, NN= rk45_solve(competition, np.array([N1_eq, N2_eq]), 1.0, 50.0, (a, b, c, d))
print("Test 3:", N1_eq, N2_eq, "->", NN[-1, 0], NN[-1, 1])

# Test 4: predator-prey should stay at N1=c/d, N2=a/b
a, b, c, d = 1, 2, 1, 3
tt, NN= rk45_solve(predprey, np.array([c / d, a / b]), 0.05, 50.0, (a, b, c, d))
print("Test 4:", c / d, a / b, "->", NN[-1, 0], NN[-1, 1])

#Question 1 
N_start = np.array([0.3, 0.6])
coeffs = (1, 2, 1, 3)               # a, b, c, d from the lab

#competition: Euler with dt = 1 yr, and rk45
t_ce, N_ce= euler_solve(competition, N_start, 1.0, 100.0, coeffs)
t_cr, N_cr= rk45_solve(competition, N_start, 1.0, 100.0, coeffs)

#predator-prey: Euler with dt = 0.05 yr, and rk45
t_pe, N_pe= euler_solve(predprey, N_start, 0.05, 100.0, coeffs)
t_pr, N_pr= rk45_solve(predprey, N_start, 0.05, 100.0, coeffs)

# Figure 1: solid = rk45, dashed = Euler
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t_cr, N_cr[:, 0], 'b-', label='N1 rk45')
plt.plot(t_cr, N_cr[:, 1], 'r-', label='N2 rk45')
plt.plot(t_ce, N_ce[:, 0], 'b--', label='N1 Euler')
plt.plot(t_ce, N_ce[:, 1], 'r--', label='N2 Euler')
plt.title('Competition')
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_pr, N_pr[:, 0], 'b-', label='Prey rk45')
plt.plot(t_pr, N_pr[:, 1], 'r-', label='Predator rk45')
plt.plot(t_pe, N_pe[:, 0], 'b--', label='Prey Euler')
plt.plot(t_pe, N_pe[:, 1], 'r--', label='Predator Euler')
plt.title('Predator-prey')
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.legend()

plt.tight_layout()
plt.savefig('fig1_euler_vs_rk45.png')

#Figure 2: What happens to Euler when I change the time step?
plt.figure(figsize=(12, 4))
for dt_try in [0.1, 0.05, 0.01]:
    t_try, N_try = euler_solve(predprey, N_start, dt_try, 100.0, coeffs)
    plt.subplot(1, 2, 1)
    plt.plot(t_try[t_try <= 20], N_try[t_try <= 20, 0], label='Euler dt = ' + str(dt_try))
    plt.subplot(1, 2, 2)
    # only plot the first 20 years so the dt = 0.1 curve does not blow up the axes
    plt.plot(N_try[t_try <= 20, 0], N_try[t_try <= 20, 1], label='Euler dt = ' + str(dt_try))

#add rk45 on top for comparison
plt.subplot(1, 2, 1)
plt.plot(t_pr[t_pr <= 20], N_pr[t_pr <= 20, 0], 'k-', label='rk45')
plt.title('Prey, first 20 years')
plt.xlabel('Time (years)')
plt.ylabel('Prey population')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(N_pr[t_pr <= 20, 0], N_pr[t_pr <= 20, 1], 'k-', label='rk45')
plt.title('Phase diagram, first 20 years')
plt.xlabel('Prey population')
plt.ylabel('Predator population')
plt.legend()

plt.tight_layout()
plt.savefig('fig2_euler_timestep.png')

#numbers to quote in the report
print("Q1 competition final: Euler", N_ce[-1], " rk45", N_cr[-1])
print("Q1 predator-prey max prey: Euler", N_pe[:, 0].max(), " rk45", N_pr[:, 0].max())

#extra check: Euler (dt = 1) vs rk45 for the competition model, other starts
for start in [(0.6, 0.3), (0.1, 0.1), (0.9, 0.05)]:
    tt, NE= euler_solve(competition, np.array(start), 1.0, 100.0, (1, 2, 1, 3))
    tt, NR= rk45_solve(competition, np.array(start), 1.0, 100.0, (1, 2, 1, 3))
    print(start, "Euler", NE[-1], "rk45", NR[-1])

#Euler with dt = 1 when the growth rates are bigger (a = 3, c = 4)
tt, NE= euler_solve(competition, np.array([0.1, 0.1]), 1.0, 100.0, (3, 2, 4, 3))
print("Euler with a=3, c=4:", NE[-1])      # should prints -inf (with an overflow warning)

#Question 2
cases = [
    ['A: a=1,b=2,c=1,d=3, start (0.3,0.6)',      1, 2,   1, 3,   0.3, 0.6],
    ['B: a=1,b=2,c=1,d=3, start (0.6,0.3)',      1, 2,   1, 3,   0.6, 0.3],
    ['C: a=1,b=0.5,c=1,d=0.5, start (0.1,0.8)',  1, 0.5, 1, 0.5, 0.1, 0.8],
    ['D: a=3,b=2,c=4,d=3, start (0.1,0.1)',      3, 2,   4, 3,   0.1, 0.1],
    ['E: a=1,b=2,c=1,d=0.5, start (0.5,0.5)',    1, 2,   1, 0.5, 0.5, 0.5],
    ['F: a=1,b=0.5,c=1,d=0.5, start (0.9,0.05)', 1, 0.5, 1, 0.5, 0.9, 0.05],
]

plt.figure(figsize=(14, 7))
for i in range(len(cases)):
    # pull the numbers out of the list
    label = cases[i][0]
    a, b, c, d = cases[i][1], cases[i][2], cases[i][3], cases[i][4]
    start = np.array([cases[i][5], cases[i][6]])

    #solve it
    t_c, N_c = rk45_solve(competition, start, 0.1, 30.0, (a, b, c, d))

    #plot it in box number i+1 of a 2 by 3 grid
    plt.subplot(2, 3, i + 1)
    plt.plot(t_c, N_c[:, 0], 'b-', label='N1')
    plt.plot(t_c, N_c[:, 1], 'r-', label='N2')
    plt.title(label, fontsize=8)
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.legend()

    print(label, "final", round(N_c[-1, 0], 3), round(N_c[-1, 1], 3))

plt.tight_layout()
plt.savefig('fig3_competition_cases.png')

#Question 3
pp_cases = [
    ['Start at equilibrium (1/3, 0.5)',   1, 2, 1, 3,   1 / 3, 0.5],
    ['Start near equilibrium (0.4, 0.5)', 1, 2, 1, 3,   0.4, 0.5],
    ['Start far away (0.9, 0.1)',         1, 2, 1, 3,   0.9, 0.1],
    ['a = 2',                             2, 2, 1, 3,   0.3, 0.6],
    ['c = 2',                             1, 2, 2, 3,   0.3, 0.6],
    ['b = 4',                             1, 4, 1, 3,   0.3, 0.6],
]

#Figure 4: population vs time
plt.figure(figsize=(14, 7))
for i in range(len(pp_cases)):
    label = pp_cases[i][0]
    a, b, c, d = pp_cases[i][1], pp_cases[i][2], pp_cases[i][3], pp_cases[i][4]
    start = np.array([pp_cases[i][5], pp_cases[i][6]])

    t_p, N_p= rk45_solve(predprey, start, 0.05, 40.0, (a, b, c, d))

    plt.subplot(2, 3, i + 1)
    plt.plot(t_p, N_p[:, 0], 'b-', label='Prey')
    plt.plot(t_p, N_p[:, 1], 'r-', label='Predator')
    plt.title(label, fontsize=9)
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.legend()
plt.tight_layout()
plt.savefig('fig4_predprey_time.png')

#Figure 5: phase diagrams (prey on x axis, predator on y axis)
plt.figure(figsize=(14, 7))
for i in range(len(pp_cases)):
    label = pp_cases[i][0]
    a, b, c, d = pp_cases[i][1], pp_cases[i][2], pp_cases[i][3], pp_cases[i][4]
    start = np.array([pp_cases[i][5], pp_cases[i][6]])

    t_p, N_p= rk45_solve(predprey, start, 0.05, 40.0, (a, b, c, d))

    plt.subplot(2, 3, i + 1)
    plt.plot(N_p[:, 0], N_p[:, 1], 'k-')
    plt.plot(N_p[0, 0], N_p[0, 1], 'go', label='start')
    plt.plot(c / d, a / b, 'r*', markersize=12, label='equilibrium')
    plt.title(label, fontsize=9)
    plt.xlabel('Prey population')
    plt.ylabel('Predator population')
    plt.legend()
plt.tight_layout()
plt.savefig('fig5_predprey_phase.png')

#Question 4
#three different starting points to try
starts = [np.array([1.0, 1.0, 1.0]),
          np.array([5.0, -3.0, 10.0]),
          np.array([-8.0, 4.0, 20.0])]

for r_value in [23, 25]:
    plt.figure(figsize=(14, 7))

    for j in range(len(starts)):
        X0 = starts[j]
        X0_pert = X0 + 1e-5          # add 1e-5 to x, y and z

        #solve the original and the perturbed run
        t_l, X_l= rk45_solve(lorenz, X0, 0.01, 100.0, (10.0, r_value, 8.0 / 3.0))
        t_l, X_p= rk45_solve(lorenz, X0_pert, 0.01, 100.0, (10.0, r_value, 8.0 / 3.0))

        #distance between the two runs at every time
        diff=np.sqrt(np.sum((X_l - X_p) ** 2, axis=1))

        #top row: x vs time for both runs
        plt.subplot(2, 3, j + 1)
        plt.plot(t_l, X_l[:, 0], 'b-', label='original')
        plt.plot(t_l, X_p[:, 0], 'r--', label='perturbed')
        plt.title('r = ' + str(r_value) + ', start = ' + str(X0), fontsize=9)
        plt.xlabel('Time')
        plt.ylabel('x')
        plt.legend()

        #bottom row: distance between runs (log scale so growth is a straight line)
        plt.subplot(2, 3, j + 4)
        plt.semilogy(t_l, diff + 1e-16, 'k-')
        plt.xlabel('Time')
        plt.ylabel('Distance between runs')

        print("r", r_value, "start", X0, "diff at t=100", diff[-1], "max", diff.max())

    plt.tight_layout()
    plt.savefig('fig6_lorenz_r' + str(r_value) + '.png')

# the butterfly shape, x vs z at r = 25
t_l, X_l= rk45_solve(lorenz, np.array([1.0, 1.0, 1.0]), 0.01, 100.0, (10.0, 25, 8.0 / 3.0))
plt.figure(figsize=(6, 5))
plt.plot(X_l[:, 0], X_l[:, 2], 'k-', linewidth=0.5)
plt.title('Lorenz attractor, r = 25')
plt.xlabel('x')
plt.ylabel('z')
plt.savefig('fig7_lorenz_attractor.png')

plt.show()      #show all the figures on screen