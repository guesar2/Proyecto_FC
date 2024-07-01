import numpy as np
import scipy


class IsingModel:
    def __init__(self, size, kb, J, H, n_mcsteps, n_eqsteps=100000):
        self.size = size
        self.lattice = np.ones((size, size))
        self.kb = kb
        self.J = J
        self.H = H
        self.n_mcsteps = n_mcsteps
        self.n_eqsteps = n_eqsteps

    def calculate_energy_change(self, i, j):
        spin = self.lattice[i, j]
        neighbors_sum = (
            self.lattice[(i + 1) % self.size, j]
            + self.lattice[i, (j + 1) % self.size]
            + self.lattice[(i - 1) % self.size, j]
            + self.lattice[i, (j - 1) % self.size]
        )
        dE = -2 * self.H * spin + 2 * self.J * spin * neighbors_sum
        return dE

    def update(self, T):
        i, j = np.random.randint(self.size, size=2)
        dE = self.calculate_energy_change(i, j)

        if np.random.rand() < np.exp(dE / (self.kb * T)):
            dM = -2 * self.lattice[i, j]
            self.lattice[i, j] *= -1
            return dE, dM
        else:
            return 0, 0

    def calculate_energy(self):
        kernel = scipy.ndimage.generate_binary_structure(2, 1)
        kernel[1][1] = False
        E_0 = self.lattice * scipy.ndimage.convolve(
            self.lattice, kernel, mode="constant"
        )

        return E_0.sum()

    def calculate_magnetization(self):
        return np.sum(self.lattice)

    def thermalize(self, T):
        energies = np.zeros(self.n_eqsteps)
        magnetizations = np.zeros(self.n_eqsteps)
        energies[0] = self.calculate_energy()
        magnetizations[0] = self.calculate_magnetization()

        for i in range(1, self.n_eqsteps):
            dE, dM = self.update(T)
            energies[i] = energies[i - 1] + dE
            magnetizations[i] = magnetizations[i - 1] + dM

        return energies[-1], magnetizations[-1]

    def run_mc(self, Ti, Tf):
        temperatures = np.linspace(Ti, Tf, self.n_mcsteps)
        temperatures = np.concatenate((temperatures, temperatures[-2::-1]))
        energies = np.zeros(2 * self.n_mcsteps - 1)
        magnetizations = np.zeros(2 * self.n_mcsteps - 1)

        for i in range(2 * self.n_mcsteps - 1):
            if i < self.n_mcsteps:
                energies[i], magnetizations[i] = self.thermalize(temperatures[i])
            else:
                energies[i], magnetizations[i] = self.thermalize(
                    temperatures[2 * self.n_mcsteps - i - 1]
                )

        return temperatures, energies, magnetizations


if __name__ == "__main__":
    model = IsingModel(size=5, kb=1, J=1, H=0, n_mcsteps=10)
    model.run_mc(.15, 10)
