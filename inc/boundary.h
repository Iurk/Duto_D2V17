#ifndef __BOUNDARY_H
#define __BOUNDARY_H

void inlet_BC(double, double*, double*, double*, double*, double*, double*, double*);
void outlet_BC(double, double*);
void bounce_back(double*);

#endif