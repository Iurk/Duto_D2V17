#ifndef __SYSSOLVE_H
#define __SYSSOLVE_H

#include <nvector/nvector_serial.h>

#define NVAR	4

void solving(unsigned int, double*, double*, void*, int (*function)(N_Vector, N_Vector, void*));
#endif