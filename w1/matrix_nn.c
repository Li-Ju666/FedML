#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN_NODE 2
#define BATCH_SIZE 2
#define EPOCH 10000

double* read_file(const char*, int*, int*); 
void vis(double*, int, int); 
double* multiply(double*, int, int, double*, int, int); 
void init_matrix(double*, int, int); 
void shuffle(double*, double*, int, 
        double*, double*, int, 
        int, int); 

int main(int argc, char** argv){
    if(argc != 3){
        printf("Invalid input: 2 parameters required! \n"); 
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
    
    /* printf("Training set: x\n"); */
    /* vis(training_x, num_sample, num_feature); */
    /* printf("Training set: y\n"); */
    /* vis(training_y, num_sample, num_1); */
    /* double A[8] = {1,2,3,4,5,6,7,8}; */
    /* double B[4] = {2,3,4,5}; */
    /* double* C = multiply(A, 4, 2, B, 2, 2); */
    /* vis(C, 4, 2); */
    /* free(C); */
    
    double* input_x = malloc(sizeof(double)*BATCH_SIZE*num_feature); 
    double* input_y = malloc(sizeof(double)*BATCH_SIZE*num_target); 
    for(int i=0; i<10; i++){
	shuffle(input_x, input_y, BATCH_SIZE, training_x,training_y, num_sample, num_feature, num_target);
	printf("i = %d: \nInput matrix X is: \n", i);
	vis(input_x, BATCH_SIZE, num_feature);
	printf("Matrix Y is: \n");
	vis(input_y, BATCH_SIZE, num_target);
    }
    double* hidden_weight = malloc(sizeof(double)*num_feature*HIDDEN_NODE); 
    double* hidden_bias = malloc(sizeof(double)*HIDDEN_NODE); 
    double* output_weight = malloc(sizeof(double)*HIDDEN_NODE*num_target); 
    double* output_bias = malloc(sizeof(double)*num_target); 
    
    init_matrix(hidden_weight, num_feature, HIDDEN_NODE); 
    init_matrix(hidden_bias, HIDDEN_NODE, 1); 
    init_matrix(output_weight, HIDDEN_NODE, num_target); 
    init_matrix(output_bias, num_target, 1); 

    double* hidden_layer, output_layer; 

    for(int iteration; iteration<EPOCH*num_sample/BATCH_SIZE; iteration++){
	shuffle(input_x, input_y, BATCH_SIZE,
	    training_x, training_y, num_sample,
	num_feature, num_target);
	



    }


    free(hidden_weight); 
    free(hidden_bias); 
    free(output_weight); 
    free(output_bias); 
    free(training_x); 
    free(training_y); 
    free(input_x); 
    free(input_y); 
    return 0; 
}

double* read_file(const char* file, int* dim1, int* dim2){
    FILE* fp; 
    if((fp = fopen(file, "r")) == NULL){
        perror("Could not open file! \n"); 
        return NULL; 
    }
    fscanf(fp, "%d ", dim1); 
    fscanf(fp, "%d ", dim2); 
    double* data = malloc(sizeof(double)*(*dim1)*(*dim2)); 
    for(int i=0; i<(*dim1)*(*dim2); i++){
        fscanf(fp, "%lf ", &(data[i])); 
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
        memcpy(&(input_x[i*num_feature]), &(all_x[temp*num_feature]), num_feature*sizeof(double)); 
        memcpy(&(input_y[i*num_target]), &(all_y[temp*num_target]), num_target*sizeof(double)); 
    }
}
