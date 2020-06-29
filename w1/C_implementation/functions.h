#include <stdio.h>
/* #include <list> */
#include <stdlib.h>
#include <math.h>

#define FEATURE 2
#define HIDDEN_NODES 2
#define OUTPUT 1

static const int num_input = FEATURE; 
static const int num_hidden_nodes = HIDDEN_NODES; 
static const int num_output = OUTPUT; 

/* Activation functions */
double sigmoid(double); 
double dsigmoid(double); 

/* Initialize weight - generate random doubles */
double init_weight(); 

double sigmoid(double x){
    return 1/(1+exp(-x)); 
}

double dsigmoid(double x){
    return x*(1-x); 
}

double init_weight(){
    return ((double)rand()/(double)RAND_MAX); 
}

void shuffle(int* array, size_t n){
    if(n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
