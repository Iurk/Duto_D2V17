#ifndef __BOUNDARY_H
#define __BOUNDARY_H

void inlet_BC(double, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void outlet_BC(double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, std::string);
void bounce_back(double*);
void wall_velocity(double*, double*, double*, double*, double*, double*, double*, double*, double*);

#endif