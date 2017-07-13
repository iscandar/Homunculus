#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* import homunculus library */
#include "homunculus.h"

int main()
{
	/*files names*/
    char brain_setting[]="example/brain_created/brain_or.data";
    char dataset[]="example/dataset/or.data";
    char new_dataset[]="example/dataset_normalized/or.data";
    /* array of neuros per layer*/
    int hidden_neurons[1] ={8};
    /*random input for testing*/
    double input[2]={0,1};
    /*creation of homunculus brain
    ** brain_init(num of inputs, num of layers, array of neuros per layer, num of output)*/
    homunculus_brain* brain= brain_init(2,1,hidden_neurons,1);
    /* some print to see some information about new brain */
    printf("Hello! this is a program to create an atificial neural network\n");
    printf("numero di input: %d \n",(int)brain -> num_inputs);
    printf("layer input neurons: %d \n",brain -> layer_input -> num_neurons);
    printf("layer nascosti: %d \n",brain ->num_hidden_layers);
    
    /*normalization of data with 
    **normalize_dataset (old dataset, new dataset, brain range)*/
    
    normalize_dataset(dataset,new_dataset,brain -> range);
    /* training with 
    **brain_training( brain, dataset normalized name, file name where we save brain setting, **
    ** learning rate, momentum, error accepted,epoche)*/
    run_training(brain,new_dataset,brain_setting,0.8,0.3,0.001,10000);
    /*liberate memory*/
    homunculus_brain_free(brain);
    /*load settings*/
    homunculus_brain* second_brain =load_setting(brain_setting);
    /*test random*/
    double *temp = run_brain(brain,input);
    double revers_normale=inverse_normalize_data(brain -> range,temp[0],0,1.0);
    printf("\nrandom result waited 1 and is %lf \n test finish.", revers_normale);
    
    free(temp);
    homunculus_brain_free(second_brain);
    return 0;
}
