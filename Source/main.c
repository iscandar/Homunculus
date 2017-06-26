#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "homunculus.h"

int main()
{
    char file1[]="input.data";//nome file di input
    char file2[]="desidered.data";//nume file dei valori desiderati
    char file3[]="data_brain.data";
    char file4[]="data_second_brain.data";
    int hidden_neurons[1] ={9,4};//,36,45,81,9};//vettore dei layer nascosti e la loro rispettiva quantita di neuroni

    _TYPE_PRECISION** input=load_matrix(file1);//caricamento primo file
    _TYPE_PRECISION** desiderato=load_matrix(file2);//caricamento secondo file
    //creazione della rete
    //numero di input,numero di hiddenlayer,vettore nei neuroni per ogni layer, output
    homunculus_brain* brain= brain_init(2,1,hidden_neurons,1);

    //test della rete
    /*printf("Hello world! this is a program to create an atificial neural network\n");
    printf("numero di input: %d \n",(int)brain -> num_inputs);
    printf("layer input neurons: %d \n",brain -> layer_input -> num_neurons);
    printf("layer nascosti: %d \n",brain ->num_hidden_layers);
    run_train(brain,input,100,desiderato,file3);
    see_brain(brain);
    homunculus_brain_free(brain);*/
    homunculus_brain* second_brain =load_setting(file3);
    see_brain(second_brain);
    run_brain(second_brain,input);
     homunculus_brain_free(second_brain);
    free(input);
    free(desiderato);

    return 0;
}

