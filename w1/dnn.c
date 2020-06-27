#include "functions.h"

#define FEATURE 2
#define HIGGEN_NODES 2
#define OUTPUT 1
#define NUM_TRAIN 4; 

int main(){
    /* define DNN architecture */
    static const int num_input = FEATURE; 
    static const int num_hidden_nodes = HIDDEN_NODES; 
    static const int num_output = OUTPUT; 

    double hidden_layer[num_hidden_nodes]; 
    double output_layer[num_output]; 

    double hidden_weights[num_input][num_hidden_nodes]; 
    double output_weights[num_hidden_nodes][num_output]; 

    /* define training data */
    static const int num_training = NUM_TRAIN; 
    double training_input[num_training][num_input] = {
	{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    }; 
    double training_output[num_training][num_output] = {
	{0.0f}, {1.0f}, {1.0f}, {0.0f}
    }; 

    int epochs = 1000; 
    for (int i=0; i<epochs; i++){
        int trainingsetorder[]={0,1,2,3}; 
        shuffle(trainingsetorder, num_training); 
        for(int x=0; x<num_training; x++){
            int i=trainingsetorder[x]; 
        }
    }
}
