#include <stdio.h>
#include <stdlib.h>
#include "homunculus.h"

int main()
{

    int hidden_neurons[2] ={2,2};
   int** input;
double desiderato[4]={1,1,0,1};

input = (int**)malloc(sizeof(int*)*4);
*input=(int*)malloc(sizeof(int)*2);
input[0][0]=1 ;
input[0][1]=0 ;
*(input+1)=(int*)malloc(sizeof(int)*2);
input[1][0]=0 ;
input[1][1]=1 ;
*(input+2)=(int*)malloc(sizeof(int)*2);
input[2][0]=0 ;
input[2][1]=0 ;
*(input+3)=(int*)malloc(sizeof(int)*2);
input[3][0]=1 ;
input[3][1]=1 ;


    homunculus_brain* brain= brain_init(2,2,hidden_neurons,1);


    printf("Hello world! this is a program to create an atificial neural network\n");
    printf("numero di input: %d \n",(int)brain -> num_inputs);
    printf("layer input neurons: %d \n",brain -> layer_input -> num_neurons);
    printf("layer nascosti: %d \n",brain ->num_hidden_layers);
    run_train(brain,input,100,desiderato);
    homunculus_brain_free(brain);
    return 0;
}

