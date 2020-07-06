#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* #define HIDDEN_NODE 10 */
#define BATCH_SIZE 2
#define EPOCH 100000
#define LEARNING_RATE 0.1f
#define WRITE_PREDICTION 1

double* read_file(const char*, int*, int*); 
void vis(double*, int, int); 
double* multiply(double*, int, int, double*, int, int); 
void init_matrix(double*, int, int); 
void shuffle(double*, double*, int, 
        double*, double*, int, 
        int, int); 
void matrix_expand(double*, int, int, double*); 
double sigmoid(double); 
double dsigmoid(double); 
void matrix_apply(double*, int, int, double(*function)(double)); 
double add_1(double a){return a+1;}; 
double quad_error(double*, double*, int, int); 
double* last_layer_delta(double*, double*, int, int); 
double* delta_to_delta(double*, double*, double*, int, int, int); 
double* trans(double*, int, int); 
void update_bias(double*, double*, double, int, int); 
void update_weight(double*, double*, double*, double, int, int, int);
void write_error(char*, double*, int); 

int main(int argc, char** argv){
    if(argc != 4){
        printf("Invalid input: 3 parameters required! \n"); 
        return -1; 
    }
    int num_sample1, num_feature, num_sample2, num_target; 
    double* training_x = read_file(argv[1], &num_sample1, &num_feature); 
    double* training_y = read_file(argv[2], &num_sample2, &num_target); 
    if(num_sample1 != num_sample2){
        perror("Invalid training dataset: samples in matrix x must equal to samples in y! \n"); 
        return -1; 
    }
    int num_sample = num_sample1; 
    double lr = LEARNING_RATE;  
    int HIDDEN_NODE = atoi(argv[3]); 
    /* printf("Training set: x\n"); */
    /* vis(training_x, num_sample, num_feature); */
    /* printf("Training set: y\n"); */
    /* vis(training_y, num_sample, num_target); */
    /* double A[8] = {1,2,3,4,5,6,7,8}; */
    /* double B[4] = {2,3,4,5}; */
    /* double* C = multiply(A, 4, 2, B, 2, 2); */
    /* vis(C, 4, 2); */
    /* free(C); */
    /* printf("A:\n"); */ 
    /* vis(A, 2, 4); */ 
    /* printf("B:\n"); */ 
    /* vis(B, 1, 4); */ 
    /* printf("Result:\n"); */ 
    /* matrix_expand(A, 2, 4, B); */ 
    /* vis(A, 2, 4); */ 
    /* printf("Add 1 result:\n") ; */ 
    /* matrix_apply(A, 2, 4, add_1); */ 
    /* vis(A, 2, 4); */ 
    /* FILE* fp = fopen("./error.txt", "w"); */ 
    double* input_x = malloc(sizeof(double)*BATCH_SIZE*num_feature); 
    double* input_y = malloc(sizeof(double)*BATCH_SIZE*num_target); 
    /* for(int i=0; i<10; i++){ */
	/* shuffle(input_x, input_y, BATCH_SIZE, training_x,training_y, num_sample, num_feature, num_target); */
	/* printf("i = %d: \nInput matrix X is: \n", i); */
	/* vis(input_x, BATCH_SIZE, num_feature); */
	/* printf("Matrix Y is: \n"); */
	/* vis(input_y, BATCH_SIZE, num_target); */
    /* } */
    double* hidden_weight = malloc(sizeof(double)*num_feature*HIDDEN_NODE); 
    double* hidden_bias = malloc(sizeof(double)*1*HIDDEN_NODE); 
    double* output_weight = malloc(sizeof(double)*HIDDEN_NODE*num_target); 
    double* output_bias = malloc(sizeof(double)*1*num_target); 
    
    double* output_delta; 
    double* hidden_delta; 

    init_matrix(hidden_weight, num_feature, HIDDEN_NODE); 
    init_matrix(hidden_bias, 1, HIDDEN_NODE); 
    init_matrix(output_weight, HIDDEN_NODE, num_target); 
    init_matrix(output_bias, 1, num_target); 

    double* hidden_layer, *output_layer; 

    double* error_his = malloc(sizeof(double)*EPOCH*num_sample/BATCH_SIZE);
    /* if(error_his != NULL){printf("Memory allocation succeed! \n"); } */ 
    for(int iteration=0; iteration<EPOCH*num_sample/BATCH_SIZE; iteration++){
    /* for(int iteration; iteration<10; iteration++){ */
	/* printf("===================\n"); */
	shuffle(input_x, input_y, BATCH_SIZE,
	    training_x, training_y, num_sample,
	num_feature, num_target);
	/* printf("Input: \n"); */ 
	/* vis(input_y, BATCH_SIZE, num_feature); */ 	
	/* Forward pass */	
	/* Input to hidden layer */
	hidden_layer = multiply(input_x, BATCH_SIZE, num_feature, 
				hidden_weight, num_feature, HIDDEN_NODE); 
	matrix_expand(hidden_layer, BATCH_SIZE, HIDDEN_NODE, hidden_bias); 
	matrix_apply(hidden_layer, BATCH_SIZE, HIDDEN_NODE, sigmoid); 

	/* Input to output layer */
	output_layer = multiply(hidden_layer, BATCH_SIZE, HIDDEN_NODE, 
				output_weight, HIDDEN_NODE, num_target); 
	matrix_expand(output_layer, BATCH_SIZE, num_target, output_bias); 
	matrix_apply(output_layer, BATCH_SIZE, num_target, sigmoid); 
	
	/* printf("Hidden weight:\n"); */ 
	/* vis(hidden_weight, num_feature, HIDDEN_NODE); */ 
	/* printf("Output weight:\n"); */ 
	/* vis(output_weight, HIDDEN_NODE, num_target); */ 	
	/* printf("Hidden bias:\n"); */ 
	/* vis(hidden_bias, 1, HIDDEN_NODE); */ 
	/* printf("Output bias:\n"); */ 
	/* vis(output_bias, 1, num_target); */ 
	/* printf("Output:\n"); */ 
	/* vis(output_layer, BATCH_SIZE, num_target); */ 
	/* printf("Expected:\n"); */ 
	/* vis(input_y, BATCH_SIZE, num_target); */ 
	/* Back propagation */	
	double error = quad_error(output_layer, input_y, BATCH_SIZE, num_target)/(BATCH_SIZE*num_target); 
	/* printf("going to save error! \n"); */ 
	error_his[iteration] = error; 
	/* printf("Succeed to save error! \n"); */ 
	output_delta = last_layer_delta(output_layer,input_y, BATCH_SIZE, num_target); 
	hidden_delta = delta_to_delta(output_delta, output_weight, hidden_layer, 
				    num_target, BATCH_SIZE, HIDDEN_NODE); 
	/* printf("Hidden delta: \n"); */ 
	/* vis(hidden_delta, BATCH_SIZE, HIDDEN_NODE); */ 
	/* printf("Output delta: \n"); */ 
	/* vis(output_delta, BATCH_SIZE, num_target); */ 
	/* Update weights and bias */
	update_bias(output_bias, output_delta, lr, BATCH_SIZE, num_target);
	update_weight(output_weight, output_delta, hidden_layer, lr, 
		    BATCH_SIZE, HIDDEN_NODE, num_target); 
	update_bias(hidden_bias, hidden_delta, lr, BATCH_SIZE, HIDDEN_NODE); 
	update_weight(hidden_weight, hidden_delta, input_x, lr, 
		    BATCH_SIZE, num_feature, HIDDEN_NODE); 
	free(hidden_layer); 
	free(output_layer); 
	free(output_delta); 
	free(hidden_delta);
    }
    
    printf("Hidden weight:\n"); 
    vis(hidden_weight, num_feature, HIDDEN_NODE);
    printf("Hidden bias:\n"); 
    vis(hidden_bias, 1, HIDDEN_NODE);  
    printf("Output weight:\n"); 
    vis(output_weight, HIDDEN_NODE, num_target); 
    printf("Output bias:\n"); 
    vis(output_bias, 1, num_target); 

#if WRITE_PREDICTION
    /* double* result = multiply(training_x, ) */

#endif   
 
    char* s1 = argv[3]; 
    char* s2 = "error_"; 
    char* s3 = malloc(strlen(s1)+strlen(s2)+1); 
    strcpy(s3, s2); 
    strcat(s3, s1); 

    write_error(s3, error_his, EPOCH*num_sample/BATCH_SIZE); 
    free(s3); 
    free(hidden_weight); 
    free(hidden_bias); 
    free(output_weight); 
    free(output_bias); 
    free(training_x); 
    free(training_y); 
    free(input_x); 
    free(input_y); 
    free(error_his); 
    return 0; 
}

double* read_file(const char* file, int* dim1, int* dim2){
    FILE* fp; 
    if((fp = fopen(file, "r")) == NULL){
        perror("Could not open file! \n"); 
        return NULL; 
    }
    if(fscanf(fp, "%d ", dim1) == EOF){return NULL; }
    if(fscanf(fp, "%d ", dim2) == EOF){return NULL; } 
    double* data = malloc(sizeof(double)*(*dim1)*(*dim2)); 
    for(int i=0; i<(*dim1)*(*dim2); i++){
        if(EOF == fscanf(fp, "%lf ", &(data[i]))){
	    return NULL; 
	} 
    }

    if(0 != fclose(fp)){
        perror("Warning: Could not close file! \n"); 
    }
    return data; 
}

void vis(double* data, int dim1, int dim2){
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%lf ", data[i*dim2+j]); 
        }
        printf("\n"); 
    }
}

double* multiply(double* A, int A1, int A2, double* B, int B1, int B2){
    if(A2 != B1){
        perror("Invalid matrix A and B: cannot be multiplied! \n"); 
        return NULL; 
    }
    double* C = calloc(sizeof(double), A1*B2); 
    double* temp = malloc(sizeof(double)*A2); 
    for(int j=0; j<B2; j++){
        for(int i=0; i<B1; i++){
            temp[i] = B[i*B2+j]; 
        }

        for(int i=0; i<A1; i++){
            for(int k=0; k<A2; k++){
                C[i*B2+j] += A[i*A2+k] * temp[k]; 
            }
        }
    }
    free(temp); 
    return C; 
}

void init_matrix(double* mat, int dim1, int dim2){
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            mat[i*dim2+j] = (double)rand()/(double)RAND_MAX; 
        }
    }
}

void shuffle(double* input_x, double* input_y, int batch, 
        double* all_x, double* all_y, int samples, 
        int num_feature, int num_target){
    int temp; 
    for(int i=0; i< batch; i++){
        temp = rand()%samples; 
        /* temp=2; */ 
	memcpy(&(input_x[i*num_feature]), &(all_x[temp*num_feature]), num_feature*sizeof(double)); 
        memcpy(&(input_y[i*num_target]), &(all_y[temp*num_target]), num_target*sizeof(double)); 
    }
}

void matrix_expand(double* A, int dim1, int dim2, double* B){
    for(int i=0; i<dim1; i++){
	for(int j=0; j<dim2; j++){
	    A[i*dim2+j] += B[j]; 
	}
    }
}

double sigmoid(double x){
    return 1/(1+exp(-x)); 
}

double dsigmoid(double x){
    return x*(1-x); 
}

void matrix_apply(double* data, int dim1, int dim2, double(*function)(double)){
    for(int i=0; i<dim1; i++){
	for(int j=0; j<dim2; j++){
	    data[i*dim2+j] = function(data[i*dim2+j]); 
	}
    }
}

double quad_error(double* A, double* B, int dim1, int dim2){
    double error=0;
    for(int i=0; i<dim1; i++){
	for(int j=0; j<dim2; j++){
	    error += 0.5*(A[i*dim2+j] - B[i*dim2+j]) * (A[i*dim2+j] - B[i*dim2+j]); 
	}
    }
    return error; 
}

double* last_layer_delta(double* output, double* expected, int dim1, int dim2){
    double* delta = calloc(sizeof(double), dim1*dim2); 
    for(int i=0; i<dim1; i++){
	for(int j=0; j<dim2; j++){
	    delta[i*dim2+j] = (output[i*dim2+j] - expected[i*dim2+j])*dsigmoid(output[i*dim2+j]); 
	}
    }
    return delta; 
}

double* delta_to_delta(double* upper_delta, double* weight, double* cur_layer, 
	int upper_dim, int batch_size, int cur_dim){
    double* delta; 
    double* trans_weight = trans(weight, cur_dim, upper_dim); 
    delta = multiply(upper_delta, batch_size, upper_dim, trans_weight, upper_dim, cur_dim); 

    for(int i=0; i<batch_size; i++){
	for(int j=0; j<cur_dim; j++){
	    delta[i*cur_dim+j] *= dsigmoid(cur_layer[i*cur_dim+j]); 
	}
    }
    free(trans_weight); 
    return delta; 
}


double* trans(double* A, int dim1, int dim2){
    double* B = malloc(sizeof(double)*dim1*dim2); 
    for(int i=0; i<dim1; i++){
	for(int j=0; j<dim2; j++){
	    B[j*dim1+i] = A[i*dim2+j]; 
	}
    }
    return B; 
}

void update_bias(double* bias, double* delta, double lr, int batch_size, int dim){
    double temp; 
    for(int j=0; j<dim; j++){
	temp = 0; 
	for(int i=0; i<batch_size; i++){
	    temp += delta[i*dim+j]; 
	}
	bias[j] -= temp/(double)batch_size*lr; 
    }
}

void update_weight(double* weight, double* delta, double* A, 
	double lr, int batch_size, int left_dim, int right_dim){
    double* trans_A = trans(A, batch_size, left_dim); 
    double* result = multiply(trans_A, left_dim, batch_size, delta, batch_size, right_dim); 
    for(int i=0; i<left_dim; i++){
	for(int j=0; j<right_dim; j++){
	    weight[i*right_dim+j] -= lr*result[i*right_dim+j]/batch_size; 
	}
    }
    free(trans_A); 
    free(result); 
}

void write_error(char* output, double* error, int num){
    FILE* fp = fopen(output, "w"); 
    /* printf("File opened! \n"); */ 
    for(int i=0; i<num; i++){
	/* printf("%f ", error[i]); */ 
	fprintf(fp, "%f ", error[i]); 
    }
    fclose(fp); 
}
