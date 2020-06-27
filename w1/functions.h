#include <stdio.h>

#define FEATURE 2
#define HIGGEN_NODES 2
#define OUTPUT 1

static const int num_input = FEATURE; 
static const int num_hidden_nodes = HIDDEN_NODES; 
static const int num_output = OUTPUT; 

/* Activation functions */
double sigmoid(double); 
double dsigmoid(double); 

/* Initialize weight - generate random doubles */
double init_weight(); 


