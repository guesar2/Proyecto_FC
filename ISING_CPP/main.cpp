#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include "spin.hpp"

int main(){
  unsigned int N = 50;
  Spin lattice(N);
  lattice.configuration_update(0.01, 1.0, 100.0, 10000).print_lattice();
  std::cout << "Energy: " << lattice.get_energy() << " " <<
               "Magnetization: " << lattice.get_magnetization() << std::endl;

  return 0;
}

