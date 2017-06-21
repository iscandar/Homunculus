#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "homunculus.h"

int main()
{
    char file1[]="input.data";
    char file2[]="desidered.data";
    int hidden_neurons[1] ={45};//,36,45,81,9};
    _TYPE_PRECISION** input=load_matrix(file1);
    _TYPE_PRECISION** desiderato=load_matrix(file2);
    //numero di input,numero di hiddenlayer,vettore nei neuroni per ogni layer, output
    homunculus_brain* brain= brain_init(9,1,hidden_neurons,1);


    printf("Hello world! this is a program to create an atificial neural network\n");
    printf("numero di input: %d \n",(int)brain -> num_inputs);
    printf("layer input neurons: %d \n",brain -> layer_input -> num_neurons);
    printf("layer nascosti: %d \n",brain ->num_hidden_layers);
    run_train(brain,input,100,desiderato);
    homunculus_brain_free(brain);
    free(input);
    free(desiderato);
    return 0;
}

