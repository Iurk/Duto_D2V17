#include <iostream>
#include "yaml-cpp/yaml.h"

#include "utilidades.h"
#include "dados.h"
#include "paths.h"

// Opening yaml file
YAML::Node config = YAML::LoadFile("./bin/dados.yml");
YAML::Node config_lattice = YAML::LoadFile("./bin/lattices.yml");

// Getting sections
const YAML::Node& domain = config["domain"];
const YAML::Node& simulation = config["simulation"];
const YAML::Node& gpu = config["gpu"];
const YAML::Node& input = config["input"];
const YAML::Node& air = config["air"];

std::string Lattice = simulation["lattice"].as<std::string>();
const YAML::Node& lattice = config_lattice[Lattice];

namespace myGlobals{

	//Domain
	unsigned int Nx = domain["Nx"].as<unsigned int>();
	unsigned int Ny = domain["Ny"].as<unsigned int>();

	//Simulation
	unsigned int NSTEPS = simulation["NSTEPS"].as<unsigned int>();
	unsigned int NSAVE = simulation["NSAVE"].as<unsigned int>();
	unsigned int NMSG = simulation["NMSG"].as<unsigned int>();
	bool meshprint = simulation["meshprint"].as<bool>();
	double erro_max = simulation["erro_max"].as<double>();

	//GPU
	unsigned int nThreads = gpu["nThreads"].as<unsigned int>();

	//Input
	double u_max = input["u_max"].as<double>();
	double rho0 = input["rho0"].as<double>();
	double Re = input["Re"].as<double>();

	//Air
	const double mi_ar = air["mi"].as<double>();

	//Lattice Info
	unsigned int ndir = lattice["q"].as<unsigned int>();
	std::vector<int> ex_vec = lattice["ex"].as<std::vector<int>>();
	std::vector<int> ey_vec = lattice["ey"].as<std::vector<int>>();
	std::string as_str = lattice["as"].as<std::string>();
	std::string w0_str = lattice["w0"].as<std::string>();
	std::string wp_str = lattice["wp"].as<std::string>();
	std::string ws_str = lattice["ws"].as<std::string>();
	std::string wt_str = lattice["wt"].as<std::string>();
	std::string wq_str = lattice["wq"].as<std::string>();

	int *ex = ex_vec.data();
	int *ey = ey_vec.data();
	double as = equation_parser(as_str);
	double w0 = equation_parser(w0_str);
	double wp = equation_parser(wp_str);
	double ws = equation_parser(ws_str);
	double wt = equation_parser(wt_str);
	double wq = equation_parser(wq_str);

	//Memory Sizes
	const size_t mem_mesh = sizeof(bool)*Nx*Ny;
	const size_t mem_size_ndir = sizeof(double)*Nx*Ny*ndir;
	const size_t mem_size_scalar = sizeof(double)*Nx*Ny;

	// Nu and Tau
	double nu = (u_max*Ny)/Re;
	const double tau = nu*(as*as) + 0.5;

	bool *walls = read_bin(walls_mesh);
	bool *inlet = read_bin(inlet_mesh);
	bool *outlet = read_bin(outlet_mesh);

}