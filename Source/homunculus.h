#ifndef HOMUNCULUS_H
#define HOMUNCULUS_H

#include <stdio.h>

#define _TYPE_PRECISION double
#define epsilon 0.9 //capacita di apprendimento
#define alfa 0.8 //considerazione dell'apprendimento passato
#define error_accepted 0.0005

typedef double (*trans_function)(_TYPE_PRECISION n);
typedef struct brain homunculus_brain;
typedef struct neuron neuron;
typedef struct sinapsi sinapsi;
typedef struct layer layer;

struct sinapsi
{
    _TYPE_PRECISION weight,delta;
    neuron *in,*out;
};

struct neuron
{
    _TYPE_PRECISION error;
    _TYPE_PRECISION trans_value;//valore di trasferimento(valore che verra passato dopo l'applicazione della funzione di trasferimento)
    _TYPE_PRECISION prop_value;//valore di propagazione(valore di attivazione, somma di tutti gli input)
    int num_in_links, num_out_links;
    sinapsi* in_links;
    sinapsi** out_links;
     _TYPE_PRECISION (*trans_function)( _TYPE_PRECISION prop_value);//funzione di trasferimento
};

struct layer
{
    int num_neurons;
    neuron* neurons;
};

 struct brain{
    int num_inputs;
    int num_hidden_layers, *num_neurons_hidden_layer, num_outputs;
    double error;
    layer *layer_input;
    layer *layer_output;
    layer *hidden_layer;


} ;

double homunculus_random();
void homunculus_brain_free(homunculus_brain *brain);
_TYPE_PRECISION* take_output(homunculus_brain *brain);
double test_brain (homunculus_brain* brain,int **input);
void see_brain (homunculus_brain* brain);
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);
void run_train (homunculus_brain* brain, int** input, int epoche,_TYPE_PRECISION* desiderato);



#endif // HOMUNCULUS_H
