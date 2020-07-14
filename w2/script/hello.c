#include <stdio.h>
#include <stdlib.h>
#include <string.h>
double* multiply(double*, double*, int); 
void vis(double*, int); 
int main(){
    double A[4] = {1, 2, 3, 4}; 
    double B[4] = {2, 3, 4, 5}; 
    double* C = multiply(A, B, 2); 
    printf("Matrix A is: \n"); 
    vis(A, 2); 
    printf("Matrix B is: \n"); 
    vis(B, 2); 
    printf("Result is: \n"); 
    vis(C, 2); 
    free(C); 
    return 0; 
}

double* multiply(double* A, double* B, int dim){
    double* C = calloc(sizeof(double),dim*dim); 
    double* temp = malloc(sizeof(double)*dim); 
    for(int j=0; j<dim; j++){
	for(int k=0; k<dim; k++){
	    temp[k] = B[k*dim+j]; 
	}
	for(int i=0; i<dim; i++){
	    for(int k=0; k<dim; k++){
		C[i*dim+j] += A[i*dim+k]*temp[k]; 
	    }
	}
    }
    free(temp); 
    return C; 
}

void vis(double* A, int dim){
    for(int i=0; i<dim; i++){
	for(int j=0; j<dim; j++){
	    printf("%.3f ", A[i*dim+j]); 
	}
	printf("\n"); 
    }
}
