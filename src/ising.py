import numpy as np
                     
class Spins:
    def __init__(self, init_state, size):
            if init_state == 0:
                self.lattice = np.random.choice([-1, 1], size=(size, size))
            elif init_state == -1:
                self.lattice = np.full((size, size), -1)
            else:
                self.lattice = np.ones((size, size))
            self.size = size
                
    def calculate_energy(self):
        neighbors_sum = (
            np.roll(self.lattice, 1, axis=1)
            + np.roll(self.lattice, -1, axis=1)
            + np.roll(self.lattice, 1, axis=0)
            + np.roll(self.lattice, -1, axis=0)
        )
        return -1 * neighbors_sum.sum()
    
    def calculate_magnetization(self):
        return np.sum(self.lattice)
    
    def update(self, i, j, T, rand):
        dE = (
        2 
        * self.lattice[i, j]
        * (
            self.lattice[(i + 1) % self.size, j]
            + self.lattice[(i - 1) % self.size, j]
            + self.lattice[i, (j + 1) % self.size]
            + self.lattice[i, (j - 1) % self.size]
        )
        )
        if rand < - dE / (T):
            self.lattice[i, j] *= -1
            dM = 2 * self.lattice[i, j]
            return dE, dM
        
        return 0, 0


def simulate_Ising(Ti, Tf, steps, size, mcsteps, thermsteps, init_state=1, J=1, kb=1):

    spins = Spins(init_state, size)        
    E = spins.calculate_energy()
    M = spins.calculate_magnetization()
    
    temperatures = np.linspace(Ti, Tf, steps)
    energies = np.zeros((steps, mcsteps))
    magnetizations = np.zeros((steps, mcsteps))
    
    rand_pos = np.random.randint(size, size=((steps, mcsteps + thermsteps, 2)))
    rands = np.log(np.random.uniform(size=(steps, mcsteps + thermsteps)))
    
    for k in range(steps):

        for l in range(mcsteps + thermsteps):
            i, j = rand_pos[k, l]

            dE, dM = spins.update(i, j, temperatures[k], rands[k, l])
            E += dE
            M += dM
            
            if l >= thermsteps:
                energies[k, l-thermsteps] = E
                magnetizations[k, l-thermsteps] = M  


    return energies, magnetizations, temperatures

