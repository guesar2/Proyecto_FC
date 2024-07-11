import numpy as np

def simulate_Ising(Ti, Tf, steps, size, mcsteps, init_state=1, J=1, kb=1):

    def calculate_energy():
        neighbors_sum = (
            np.roll(spins, 1, axis=1)
            + np.roll(spins, -1, axis=1)
            + np.roll(spins, 1, axis=0)
            + np.roll(spins, -1, axis=0)
        )
        return -J * neighbors_sum.sum()

    def calculate_dE(i, j):
        dE = (
            2 * J
            * spins[i, j]
            * (
                spins[(i + 1) % size, j]
                + spins[(i - 1) % size, j]
                + spins[i, (j + 1) % size]
                + spins[i, (j - 1) % size]
            )
        )

        return dE
    

    if init_state == 0:
        spins = np.random.choice([-1, 1], size=(size, size))
    elif init_state == 1:
        spins = np.ones((size, size))
    elif init_state == -1:
        spins = np.full((size, size), -1)
    else:
        raise ValueError("Invalid init_state value. Use 0 for random, 1 for all up, or -1 for all down.")

        
    E = calculate_energy()
    M = np.sum(spins)
    temperatures = np.linspace(Ti, Tf, steps)
    energies = np.zeros((steps, mcsteps + 1))
    magnetizations = np.zeros((steps, mcsteps + 1))
    
    rand_pos = np.random.randint(size, size=((steps, mcsteps, 2)))
    rands = np.log(np.random.uniform(size=(steps, mcsteps)))
    
    for k in range(steps):
        energies[k, 0] = E
        magnetizations[k, 0] = M

        for l in range(mcsteps):
            i, j = rand_pos[k, l]
            dE = calculate_dE(i, j)

            if rands[k, l] < - dE / (kb * temperatures[k]):
                spins[i, j] *= -1
                M += 2 * spins[i, j]
                E += dE
                
            energies[k, l + 1] = E
            magnetizations[k, l + 1] = M


    return energies, magnetizations, temperatures



