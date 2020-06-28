#include "functions.h"

#define FEATURE 2
#define HIGGEN_NODES 2
#define OUTPUT 1
#define NUM_TRAIN 4 

int main(){
    /* define DNN architecture */
    static const double lr = 0.1f; 
    static const int num_input = FEATURE; 
    static const int num_hidden_nodes = HIDDEN_NODES; 
    static const int num_output = OUTPUT; 

    double hidden_layer[num_hidden_nodes]; 
    double output_layer[num_output]; 

    double hidden_layer_bias[num_hidden_nodes]; 
    double output_layer_bias[num_output]; 

    double hidden_weights[num_input][num_hidden_nodes]; 
    double output_weights[num_hidden_nodes][num_output]; 

    /* define training data */
    static const int num_training = NUM_TRAIN; 
    double training_input[NUM_TRAIN][FEATURE] = {
	    {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    }; 
    double training_output[NUM_TRAIN][OUTPUT] = {
	    {0.0f}, {1.0f}, {1.0f}, {0.0f}
    }; 

    for(int i=0; i<num_input; i++){
        for(int j=0; j<num_hidden_nodes; j++){
            hidden_weights[i][j] = init_weight(); 
        }
    }

    for(int i=0; i<num_hidden_nodes; i++){
        hidden_layer_bias[i] = init_weight(); 
        for(int j=0; j<num_output; j++){
            output_weights[i][j] = init_weight(); 
        }
    }

    for(int i=0; i<num_output; i++){
        output_layer_bias[i] = init_weight(); 
    }

    int training_set_order[] = {0,1,2,3}; 

    int epochs = 10000; 
    for (int n=0; n<epochs; n++){
        shuffle(training_set_order, num_training); 
        for(int x=0; x<num_training; x++){
            int i=training_set_order[x]; 

            /* Forward pass */
            
            /* Input layer to hidden layer */
            for(int j=0; j<num_hidden_nodes; j++){
                double activation = hidden_layer_bias[j]; 
                for(int k=0; k<num_input; k++){
                    activation += training_input[i][k]*hidden_weights[k][j]; 
                }
                hidden_layer[j] = sigmoid(activation); 
            }

            /* Hidden layer to output layer */
            for(int j=0; j<num_output; j++){
                double activation = output_layer_bias[j]; 
                for(int k=0; k<num_hidden_nodes; k++){
                    activation += hidden_layer[x]*output_weights[k][j]; 
                }
                output_layer[j] = sigmoid(activation); 
            }
            
            printf("Input: %f %f\nOutput: %f\n", 
                    training_input[i][0], training_input[i][1], output_layer[0]); 
            /* Back propagation */
            
            /* Calculation for output layer delta */
            double delta_output[num_output]; 
            for(int j=0; j<num_output; j++){
                double error_output = (training_output[i][j] - output_layer[j]); 
                delta_output[j] = error_output*dsigmoid(output_layer[j]); 
            }

            /* Calculation for hidden layer delta */
            double delta_hidden[num_hidden_nodes]; 
            for(int j=0; j<num_hidden_nodes; j++){
                double error_hidden = 0.0f; 
                for(int k=0; k<num_output; k++){
                    error_hidden += delta_output[k]*output_weights[j][k]; 
                }
                delta_hidden[j] = error_hidden*dsigmoid(hidden_layer[j]); 
            }

            /* Update for output weight and bias */
            for(int j=0; j<num_output; j++){
                output_layer_bias[j] += delta_output[j]*lr; 
                for(int k=0; k<num_hidden_nodes; k++){
                    output_weights[k][j] += hidden_layer[k]*lr*delta_output[j]; 
                }
            }

            /* Update for hidden layer weight and bias */
            for(int j=0; j<num_hidden_nodes; j++){
                hidden_layer_bias[j] += delta_hidden[j]*lr; 
                for(int k=0; k<num_input; k++){
                    hidden_weights[k][j] += training_input[i][k]*lr*delta_hidden[j]; 
                }
            }

        }
    }

    printf("Finall hidden weights\n{\n"); 
    for(int j=0; j<num_hidden_nodes; j++){
        printf("[ "); 
        for(int k=0; k<num_input; k++){
            printf("%f ", hidden_weights[k][j]); 
        }
        printf(" ]\n"); 
    }
    printf("}\n"); 

    printf("Final hidden bias\n{\n"); 
    for(int j=0; j<num_hidden_nodes; j++){
        printf("%f ", hidden_layer_bias[j]); 
    }
    printf("\n}\n"); 

    printf("Final output weights\n{\n"); 
    for(int j=0; j<num_output; j++){
        printf("[ "); 
        for(int k=0; k<num_hidden_nodes; k++){
            printf("%f ", output_weights[k][j]); 
        }
        printf(" ]\n"); 
    }
    printf(" }\n"); 

    printf("Final output bias\n{\n"); 
    for(int j=0; j<num_output; j++){
        printf("%f ", output_layer_bias[j]); 
    }
    printf("\n}\n"); 
    return 0; 
}
