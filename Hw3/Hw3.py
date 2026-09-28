import numpy as np
import matplotlib.pyplot as plt

# Starting values
T_initial = 90   # coffee temp (C)
T_env = 20       # room temp (C)
k = 0.1          # cooling constant

def euler(dt):
    T = T_initial
    time = 0
    times = [time]
    temps = [T]
    # Keep stepping until the coffee is within 0.5 C of room temp
    while T > T_env + 0.5:
        T = T - k * (T - T_env) * dt   # Euler step
        time = time + dt
        times.append(time)
        temps.append(T)
    return times, temps

# Analytic solution
t = np.linspace(0, 60, 200)
T_exact = T_env + (T_initial - T_env) * np.exp(-k * t)
plt.plot(t, T_exact, "k", label="Analytic")

# Euler solution for different step sizes
for dt in [10, 5, 1]:
    times, temps = euler(dt)
    print("dt =", dt, "time elapsed =", times[-1])
    plt.plot(times, temps, "--", label="Euler dt = " + str(dt))

plt.xlabel("Time (min)")
plt.ylabel("Temperature (C)")
plt.legend()
plt.show()