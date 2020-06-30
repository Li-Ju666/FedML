#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* The program is used to generate test data for */
/* matrix-based neural network: y = sin(x1+x2) is */
/* the real function here. The data is consisted of*/
/* 10000 data: 100 x1 and 100 x2, nested. */ 

int main(int argc, char** argv){
    double x1=-5.0f, x2=-5.0f;
    double* x = malloc(2*100*100*sizeof(double)); 
    double* y = malloc(100*100*sizeof(double)); 
    int i=0; 
    for(x1=-5.0f; x1<5.0f; x1+=0.1f){
	for(x2=-5.0f; x2<5.0f; x2+=0.1f){
	    x[i] = x1; 
	    i++;
	    x[i] = x2; 
	    i++; 
	    y[i/2]=(sin(x1+x2)+1)/2; 
	}
    }
    FILE* fp1 = fopen("x.dat", "w"); 
    FILE* fp2 = fopen("y.dat", "w"); 
    fprintf(fp1, "%d %d ", 10000, 2); 
    fprintf(fp2, "%d %d ", 10000, 1); 
    for(int i=0; i<10000; i++){
	fprintf(fp1, "%f ", x[2*i]); 
	fprintf(fp1, "%f ", x[2*i+1]); 
	fprintf(fp2, "%f ", y[i]); 
    }
    fclose(fp1); 
    fclose(fp2); 
    free(x); 
    free(y); 
    /* printf("Now amount is: %d. \n", i); */ 
}
