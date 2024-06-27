#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include "spin.hpp"

int main(){
  unsigned int N = 10;
  Spin lattice(N);
  lattice.configuration_update(1.48, 1.0, 0.0, 10000).print_lattice();
  std::cout << "Energy: " << lattice.get_energy() << " " <<
               "Magnetization: " << lattice.get_magnetization() << std::endl;

  return 0;
}

