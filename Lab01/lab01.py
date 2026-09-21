##Lab 01 Spread of Forest Fires


#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 1 for CLaSP
410.
To reproduce the plots shown in the lab report, do this...
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Constants
nx, ny = 3, 3 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as bare patch.
prob_start = 0.0 # Chance of cell to start on fire.

#Status codes for each cell
BARE = 1
FOREST = 2
FIRE = 3

# Code from lab document
# Create an initial grid, set all values to "2

# type in our array to integers only.
forest = np.zeros([ny, nx]) + FOREST
# Set the center cell to "burning":
forest[1, 1] = FIRE

print(forest)

# Advancing the fire by one iteration
# We make a copy so we're reading from the "old" grid while writing to a "new" one
# This avoids accidentally using cells we already updated this same iteration
new_forest = forest.copy()

for i in range(nx):
    for j in range(ny):
        # Only do something if this cell is currently on fire
        if forest[j, i] == FIRE:

            # Check the neighboring cell above (only if it exists, j+1 is in bounds)
            if j + 1 < ny:
                if forest[j + 1, i] == FOREST:
                    if np.random.rand() < prob_spread:
                        new_forest[j + 1, i] = FIRE

            # Check the neighboring cell below (only if j-1 doesn't wrap around)
            if j - 1 >= 0:
                if forest[j - 1, i] == FOREST:
                    if np.random.rand() < prob_spread:
                        new_forest[j - 1, i] = FIRE

            # Check the neighboring cell to the right
            if i + 1 < nx:
                if forest[j, i + 1] == FOREST:
                    if np.random.rand() < prob_spread:
                        new_forest[j, i + 1] = FIRE

            # Check the neighboring cell to the left
            if i - 1 >= 0:
                if forest[j, i - 1] == FOREST:
                    if np.random.rand() < prob_spread:
                        new_forest[j, i - 1] = FIRE

            # This cell was burning at the start of the iteration, so now it's bare
            new_forest[j, i] = BARE

# Replace the old grid with the new updated one
forest = new_forest

print(forest)

def advance_fire(forest, prob_spread):
    """
    Given a forest grid we advance the fire by one iteration and it returns the new grid after spreading.
    """
    ny, nx = forest.shape
    new_forest = forest.copy()

    for i in range(nx):
        for j in range(ny):
            # Only do something if this cell is currently on fire
            if forest[j, i] == FIRE:

                # Check the neighboring cell above
                if j + 1 < ny:
                    if forest[j + 1, i] == FOREST:
                        if np.random.rand() < prob_spread:
                            new_forest[j + 1, i] = FIRE

                # Check the neighboring cell below
                if j - 1 >= 0:
                    if forest[j - 1, i] == FOREST:
                        if np.random.rand() < prob_spread:
                            new_forest[j - 1, i] = FIRE

                # Check the neighboring cell to the right
                if i + 1 < nx:
                    if forest[j, i + 1] == FOREST:
                        if np.random.rand() < prob_spread:
                            new_forest[j, i + 1] = FIRE

                # Check the neighboring cell to the left
                if i - 1 >= 0:
                    if forest[j, i - 1] == FOREST:
                        if np.random.rand() < prob_spread:
                            new_forest[j, i - 1] = FIRE

                # This cell was burning at the start and now it's bare
                new_forest[j, i] = BARE

    return new_forest

#Restart the grid for a clean test
forest = np.zeros([ny, nx]) + FOREST
forest[1, 1] = FIRE

print("Iteration 0:")
print(forest)

forest = advance_fire(forest, prob_spread)
print("Iteration 1:")
print(forest)

forest = advance_fire(forest, prob_spread)
print("Iteration 2:")
print(forest)

#Test with a 3x5 grid
nx_test, ny_test = 5, 3
forest2 = np.zeros([ny_test, nx_test]) + FOREST
forest2[ny_test // 2, nx_test // 2] = FIRE  # set the center cell on fire

print("3x5 grid, Iteration 0:")
print(forest2)

forest2 = advance_fire(forest2, prob_spread)
print("3x5 grid, Iteration 1:")
print(forest2)

forest2 = advance_fire(forest2, prob_spread)
print("3x5 grid, Iteration 2:")
print(forest2)

def run_simulation(forest, prob_spread):
    """
    Runs the fire simulation until there is no more fire. Returns the final grid and the number of iterations it took.
    """
    n_iterations = 0

    # Keep looping as long as there's at least one cell with a 3 (on fire)
    while np.any(forest == FIRE):
        forest = advance_fire(forest, prob_spread)
        n_iterations = n_iterations + 1

    return forest, n_iterations

forest = np.zeros([ny, nx]) + FOREST
forest[1, 1] = FIRE

final_forest, n_iter = run_simulation(forest, prob_spread)
print("Final grid:")
print(final_forest)
print("Number of iterations:", n_iter)

n_forest_survived = np.sum(final_forest == FOREST)
n_bare = np.sum(final_forest == BARE)

print("Cells still forested:", n_forest_survived)
print("Cells bare/burnt:", n_bare)

#Task 2

#Experiment 1: vary prob_spread
spread_values = np.arange(0, 1.1, 0.1)  # 0, 0.1, 0.2, ..., 1.0. Arange stops just before the number so to 1.1 otherwise 1.0 would get cutoff
survival_fractions = []

for p in spread_values:
    forest = np.zeros([ny, nx]) + FOREST
    forest[1, 1] = FIRE

    final_forest, n_iter = run_simulation(forest, p)

    n_survived = np.sum(final_forest == FOREST)
    total_cells = nx * ny
    fraction_survived = n_survived / total_cells

    survival_fractions.append(fraction_survived) #Adds result onto running list so we can plot them altogether

print("Spread probabilities:", spread_values)
print("Survival fractions:", survival_fractions)


# Plot survival vs prob_spread
plt.figure()
plt.plot(spread_values, survival_fractions, marker='o')
plt.xlabel('Probability of spread')
plt.ylabel('Fraction of forest survived')
plt.title('Forest survival vs. probability of fire spread')
plt.show()
print("PLOT CLOSED, CONTINUING")



def initialize_forest(nx, ny, prob_bare, prob_ignite):
    """
    Creates a forest grid of size ny x nx.
    Each cell is randomly bare with probability prob_bare otherwise it starts as a forest
    Any remaining forested cell is randomly set on fire with probability prob_ignite.
    """
    #Start with every cell as forest
    forest = np.zeros([ny, nx]) + FOREST

    #Loop over every cell and randomly make some cells bare
    for i in range(nx):
        for j in range(ny):
            if np.random.rand() < prob_bare:
                forest[j, i] = BARE

    #Loop over every cell again to ignite them but skips any made bare above
    for i in range(nx):
        for j in range(ny):
            if forest[j, i] == FOREST:
                if np.random.rand() < prob_ignite:
                    forest[j, i] = FIRE

    return forest
#Test on 5x5 grid 30% bare and 10% forest starts on fire
test_forest = initialize_forest(5, 5, 0.3, 0.1)
print(test_forest)

# Experiment 2: vary prob_bare
bare_values = np.arange(0, 1.1, 0.1)  # 0, 0.1, 0.2, ..., 1.0
prob_ignite = 0.1  # keep this fixed and reasonable for this experiment
survival_fractions_bare = []

for p_bare in bare_values:
    forest = initialize_forest(nx, ny, p_bare, prob_ignite)

    final_forest, n_iter = run_simulation(forest, prob_spread)

    n_survived = np.sum(final_forest == FOREST)
    total_cells = nx * ny
    fraction_survived = n_survived / total_cells

    survival_fractions_bare.append(fraction_survived)

print("Bare probabilities:", bare_values)
print("Survival fractions:", survival_fractions_bare)

# Plot survival vs prob_bare
plt.figure()
plt.plot(bare_values, survival_fractions_bare, marker='o')
plt.xlabel('Probability of bare ground')
plt.ylabel('Fraction of forest survived')
plt.title('Forest survival vs. probability of bare ground')
plt.show()
print("PLOT CLOSED, CONTINUING")


# New status code for disease model
DEAD = 0
# Reusing: IMMUNE = 1, HEALTHY = 2, SICK = 3 (same numbers, new meaning)
IMMUNE = BARE
HEALTHY = FOREST
SICK = FIRE

def advance_disease(population, prob_spread, prob_fatal):
    """
    Given a population grid, advance the disease by one iteration.
    Sick people either die (with probability prob_fatal) or survive and
    become immune. Before that happens, sick people may infect healthy
    neighbors with probability prob_spread.
    Returns the new grid after one iteration.
    """
    ny, nx = population.shape
    new_population = population.copy()

    for i in range(nx):
        for j in range(ny):
            if population[j, i] == SICK:

                # Check the neighbor above
                if j + 1 < ny:
                    if population[j + 1, i] == HEALTHY:
                        if np.random.rand() < prob_spread:
                            new_population[j + 1, i] = SICK

                # Check the neighbor below
                if j - 1 >= 0:
                    if population[j - 1, i] == HEALTHY:
                        if np.random.rand() < prob_spread:
                            new_population[j - 1, i] = SICK

                # Check the neighbor to the right
                if i + 1 < nx:
                    if population[j, i + 1] == HEALTHY:
                        if np.random.rand() < prob_spread:
                            new_population[j, i + 1] = SICK

                # Check the neighbor to the left
                if i - 1 >= 0:
                    if population[j, i - 1] == HEALTHY:
                        if np.random.rand() < prob_spread:
                            new_population[j, i - 1] = SICK

                # This person was sick at the start of the iteration.
                # They either die or survive and become immune.
                if np.random.rand() < prob_fatal:
                    new_population[j, i] = DEAD
                else:
                    new_population[j, i] = IMMUNE

    return new_population

# Quick test of the disease model
prob_fatal = 0.2  # 20% chance a sick person dies instead of recovering

population = initialize_forest(nx, ny, prob_bare, prob_ignite)

print("Day 0:")
print(population)

population = advance_disease(population, prob_spread, prob_fatal)
print("Day 1:")
print(population)

population = advance_disease(population, prob_spread, prob_fatal)
print("Day 2:")
print(population)

# Experiment 3a: vary prob_fatal (mortality rate)
fatal_values = np.arange(0, 1.1, 0.1)
prob_bare_disease = 0.0   # no vaccine for this experiment, we isolate mortality's effect
survival_fractions_fatal = []

for p_fatal in fatal_values:
    population = initialize_forest(nx, ny, prob_bare_disease, prob_ignite)

    # Keep advancing until no one is sick anymore
    while np.any(population == SICK): #Same idea as run_simulation
        population = advance_disease(population, prob_spread, p_fatal)

    n_survived = np.sum((population == HEALTHY) | (population == IMMUNE)) #Comparing two true/false arrays
    total_cells = nx * ny
    fraction_survived = n_survived / total_cells

    survival_fractions_fatal.append(fraction_survived)

print("Fatality probabilities:", fatal_values)
print("Survival fractions:", survival_fractions_fatal)

plt.figure()
plt.plot(fatal_values, survival_fractions_fatal, marker='o')
plt.xlabel('Probability of fatality')
plt.ylabel('Fraction of population survived (healthy or immune)')
plt.title('Population survival vs. disease fatality rate')
plt.show()
print("PLOT CLOSED, CONTINUING")

# Experiment 3b: vary prob_bare (early vaccine rate)
vaccine_values = np.arange(0, 1.1, 0.1)
prob_fatal_fixed = 0.3   # keep mortality rate fixed so we isolate vaccine effect
survival_fractions_vaccine = []

for p_vaccine in vaccine_values:
    population = initialize_forest(nx, ny, p_vaccine, prob_ignite)

    while np.any(population == SICK):
        population = advance_disease(population, prob_spread, prob_fatal_fixed)

    n_survived = np.sum((population == HEALTHY) | (population == IMMUNE))
    total_cells = nx * ny
    fraction_survived = n_survived / total_cells

    survival_fractions_vaccine.append(fraction_survived)

print("Vaccine probabilities:", vaccine_values)
print("Survival fractions:", survival_fractions_vaccine)

plt.figure()
plt.plot(vaccine_values, survival_fractions_vaccine, marker='o')
plt.xlabel('Early vaccine rate (initial immune fraction)')
plt.ylabel('Fraction of population survived (healthy or immune)')
plt.title('Population survival vs. early vaccine rate')
plt.show()