#ifndef HOMUNCULUS_H
#define HOMUNCULUS_H

#include <stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define _TYPE_PRECISION double



typedef struct brain homunculus_brain;
typedef struct neuron neuron;
typedef struct sinapsi sinapsi;
typedef struct layer layer;
typedef double (*trans_function)(_TYPE_PRECISION n);
typedef double (*error_function)(neuron* n,_TYPE_PRECISION desidered);


struct sinapsi
{
    _TYPE_PRECISION weight,delta;
    neuron *in,*out;
};

struct neuron
{
    _TYPE_PRECISION error;//some time its called delta
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
    int num_inputs, num_hidden_layers, *num_neurons_hidden_layer, num_outputs;
    double error;
    double (*error_function)(neuron* n, _TYPE_PRECISION desidered);
    layer *layer_input;
    layer *hidden_layer;
    layer *layer_output;

} ;

double homunculus_random();
void homunculus_brain_free(homunculus_brain *brain);
_TYPE_PRECISION* take_output(homunculus_brain *brain);
double test_brain (homunculus_brain* brain,_TYPE_PRECISION**input);
void see_brain (homunculus_brain* brain);
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);
void run_train (homunculus_brain* brain, _TYPE_PRECISION** input, int epoche,_TYPE_PRECISION** desiderato);
_TYPE_PRECISION** load_matrix(const char *file_name);


#endif // HOMUNCULUS_H
