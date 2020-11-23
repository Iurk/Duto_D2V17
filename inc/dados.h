#ifndef __DADOS_H
#define __DADOS_H

#include <vector>

namespace myGlobals{

	//Domain
	extern unsigned int Nx, Ny;

	//Simulation
	extern unsigned int NSTEPS, NSAVE, NMSG;
	extern bool computeFlowProperties, quiet, meshprint;

	//GPU
	extern unsigned int nThreads;

	//Input
	extern double u_max, rho0, Re;
	extern double nu;
	extern const double tau;

	//Lattice Info
	extern unsigned int ndir;
	extern int *ex;
	extern int *ey;
	extern double cs, w0, ws, wd;

	//Memory Sizes
	extern const size_t mem_mesh;
	extern const size_t mem_size_0dir;
	extern const size_t mem_size_n0dir;
	extern const size_t mem_size_scalar;

	extern bool *cylinder;
	extern bool *fluid;
}

#endif