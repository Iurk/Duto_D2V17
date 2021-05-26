#include <stdio.h>
#include <stdlib.h>

#include <kinsol/kinsol.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>

#include "system_solve.h"

#define FTOL 	RCONST(1.e-5)
#define STOL 	RCONST(1.e-5)

typedef struct{
	realtype initial[NVAR];
} *UserData;

void SetInitialGuess(N_Vector, UserData);
int SolveIt(void*, N_Vector, N_Vector, int, int);
int check_flag(void*, const char*, int);

void solving(double *guess, double *solution, void *user_data, int (*function)(N_Vector, N_Vector, void*)){

	// Initializing
	UserData data;
	realtype fnormtol, scsteptol;
	N_Vector u1, u, s;
	int glstr, mset, flag;
	void *kmem;
	SUNMatrix J;
	SUNLinearSolver LS;

	u1 = u = NULL;
	s = NULL;
	kmem = NULL;
	J = NULL;
	LS = NULL;
	data = NULL;

	// Setting Initial Guess
	data = (UserData)malloc(sizeof *data);
	data->initial[0] = guess[0];
	data->initial[1] = guess[1];
	//data->initial[2] = 0.0;
	//data->initial[3] = 0.0;

	// Creating vectors
	u1 = N_VNew_Serial(NEQ);
	if(check_flag((void*)u1, "N_VNew_Serial", 0)) return;
	u = N_VNew_Serial(NEQ);
	if(check_flag((void*)u, "N_VNew_Serial", 0)) return;
	s = N_VNew_Serial(NEQ);
	if(check_flag((void*)s, "N_VNew_Serial", 0)) return;

	SetInitialGuess(u1, data);	// Copying data to u1
	N_VConst(1.0, s);			// Setting scale to 1

	fnormtol = FTOL, scsteptol = STOL;	// Setting tolerances

	// Creating the solver
	kmem = KINCreate();
	if(check_flag(&flag, "KINCreate", 0)) return;

	// Setting up
	flag = KINSetUserData(kmem, user_data);
	if(check_flag(&flag, "KINSetUserData", 1)) return;
	flag = KINSetFuncNormTol(kmem, fnormtol);
	if(check_flag(&flag, "KINSetFuncNormTol", 1)) return;
	flag = KINSetScaledStepTol(kmem, scsteptol);
	if(check_flag(&flag, "KINSetScaledStepTol", 1)) return;

	// Initializing the solver, the Jacobian and the Linear Solver
	flag = KINInit(kmem, function, u);
	if(check_flag(&flag, "KINInit", 1)) return;

	J = SUNDenseMatrix(NEQ, NEQ);
	if(check_flag((void*)J, "SUNDesneMatrix", 0)) return;
	LS = SUNLinSol_Dense(u, J);
	if(check_flag((void*)LS, "SUNLinSol_Dense", 0)) return;
	flag = KINSetLinearSolver(kmem, LS, J);
	if(check_flag(&flag, "KINSetLinearSolver", 1)) return;

	N_VScale(1.0, u1, u);		// Setting scale to 1
	glstr = KIN_NONE;			// Setting the method to solve the system
	mset = 0;					// Setting the maximum number of nonlinear iterations without a call to the preconditioner or Jacobian setup function
	SolveIt(kmem, u, s, glstr, mset); // Calling the solver

	solution[0] = NV_Ith_S(u, 0);
	solution[1] = NV_Ith_S(u, 1);
	solution[2] = NV_Ith_S(u, 2);
	solution[3] = NV_Ith_S(u, 3);

	N_VDestroy(u1);
	N_VDestroy(u);
	N_VDestroy(s);
	KINFree(&kmem);
	SUNMatDestroy(J);
	SUNLinSolFree(LS);
	free(data);
}

void SetInitialGuess(N_Vector u, UserData data){

	realtype *udata;
	realtype *initial;

	udata = N_VGetArrayPointer(u);

	initial = data->initial;

	udata[0] = initial[0];
	udata[1] = initial[1];
	udata[2] = initial[2];
	udata[3] = initial[3];
}

int SolveIt(void *kmem, N_Vector u, N_Vector s, int glstr, int mset){

	int flag;

	flag = KINSetMaxSetupCalls(kmem, mset); 					// Setting the maximum number of nonlinear iterations without a call to the preconditioner or Jacobian setup function
	if(check_flag(&flag, "KINSetMaxSetupCalls", 1)) return(1);
	flag = KINSol(kmem, u, glstr, s, s);						// Solving the system
	if(check_flag(&flag, "KINSol", 1)) return(1);

	return(0);
}

int check_flag(void *flagvalue, const char *funcname, int opt){
	int *errflag;

	if(opt == 0 && flagvalue == NULL){
		fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
		return(1);
	}

	else if(opt == 1){
		errflag = (int *)flagvalue;
		if(*errflag < 0){
			fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag);
			return(1);
		}
	}

	else if(opt == 2 && flagvalue == NULL){
		fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
		return(1);
	}
	return(0);
}