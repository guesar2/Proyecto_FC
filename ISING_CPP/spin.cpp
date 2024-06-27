#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <omp.h>
#include "spin.hpp"

Spin::Spin(const unsigned int &size){
  N = size;
  lattice = std::vector<int>(size*size, 1);
  energy = 0;
}

Spin::Spin(const Spin &obj){
  N = obj.N;
  lattice = obj.lattice;
  energy = obj.energy;
}

Spin::~Spin(){

}

Spin &Spin::operator=(const Spin &obj){
  N = obj.N;
  lattice = obj.lattice;
  energy = obj.energy;

  return *this;
}

int Spin::get_N(){
  return N;
}

void Spin::print_lattice(){
  for (unsigned int i = 0; i < N; ++i){
    for (unsigned int j = 0; j < N; ++j){
      std::cout << lattice[ (i*N) + j ] << " ";
    }
    std::cout << std::endl;
  }
}

int Spin::get_magnetization(){
  int magnetization = 0;
  #pragma omp parallel reduction (+ : magnetization)
  {
    #pragma omp for
    for (unsigned int i = 0; i < N*N; ++i){
      magnetization += lattice[i];
    }
  }

  return magnetization;
}

int Spin::close_neighbord_energy(const unsigned int &row, const unsigned int &col){
  int dE = 0;
  dE = 2*lattice[ (row*N) + col ]*( lattice[ ((row-1)*N)%N + col ] + 
                                    lattice[ ((row+1)*N)%N + col ] + 
                                    lattice[ (row*N) + (col-1)%N ] + 
                                    lattice[ (row*N) + (col+1)%N ] );
  return dE;
}

double Spin::get_energy(){
  return energy;
  }

void Spin::flip(const unsigned int &row, const unsigned int &col){
  lattice[ (row*N) + col ] = (lattice[ (row*N) + col ] > 0) ? -1 : 1;
}

Spin Spin::configuration_update(const double &beta, const float &J, const float &H, const unsigned int &max_iter){
  int dE = 0;
  double dH = 0.0;

  std::random_device rd; 
  std::mt19937 rng(rd());
  std::uniform_int_distribution<unsigned int> rndint(0, N);
  std::uniform_real_distribution<double> rndb(0, 1);
  
  #pragma omp parallel reduction(+ : energy) private(dE, dH, rng, rndint, rndb)
  {
    #pragma omp for
    for (unsigned int i = 0; i < max_iter; ++i){
      unsigned int row = rndint(rng)%N;
      unsigned int col = rndint(rng)%N;
      Spin lattice_copy = *this;   
      lattice_copy.flip(row,col);

      dE = lattice_copy.close_neighbord_energy(row, col);
      dH = -H*(lattice_copy.get_magnetization() - get_magnetization()) - J*dE;
      if (rndb(rng) < exp(-beta*dH)){
        #pragma omp atomic
        energy += dE;
        *this = lattice_copy;
      }
    }
  }
  return *this;
}
