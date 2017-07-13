/****************************************************************************
 * Copyright (C) 2017 by Alejandro Ontivero                                 *
 *                                                                          *
 * This file is part of Homunculus.                                         *
 *                                                                          *
 ****************************************************************************/
 /**
 * @file homunculus.h
 * @author Alejandro Ontivero
 * @date 5 Luglio 2017
 * @brief Questo file contiene la struttura della libreria homunculus
 *
 */
#ifndef HOMUNCULUS_H
#define HOMUNCULUS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


/**
 * @brief Utilizzo di typedef per definire in maniera piu semplice tutte le strutture
 */
 #define _TYPE_PRECISION double	/**< Definizione di double come _TYPE_PRECISION per indicare la qualita di percisione nei calcoli */
 /**
 * @brief Definizioni di range per calcolo della propagazione
 */
static double sig_range[2] = {0,1}; /**< E' il campo di esistenza della sigmoide */
static double tanh_range[2] = {-1,1};/**< E' il campo di esistenza della tanh */
typedef struct brain homunculus_brain;
typedef struct neuron neuron;
typedef struct sinapsi sinapsi;
typedef struct layer layer;
typedef _TYPE_PRECISION (*trans_function)(_TYPE_PRECISION n);
typedef _TYPE_PRECISION (*error_function)(neuron* n,_TYPE_PRECISION desidered);

 /**
 * @brief Struttura del neurone
 */
struct neuron
{
    _TYPE_PRECISION error;/**< E' l'errore complessivo generato dal neurone */
    _TYPE_PRECISION trans_value;/**< E' il valore di trasferimento ottenuto dalla funzione di trasferimento */
    _TYPE_PRECISION prop_value;/**< E' la somma dei segnali in input */
    int num_in_links, num_out_links;/**< Numero di link in entrata e il numero di link in uscita */
    sinapsi* in_links;/**< Array di collegamenti */
    sinapsi** out_links;/**< Array di collegamenti */
     _TYPE_PRECISION (*trans_function)( _TYPE_PRECISION prop_value);/**< E' il puntatore alla funzione di transferimento */
};

 /**
 * @brief Struttura degli strati
 */
struct layer
{
    int num_neurons;/**< Numero di neuroni di neuroni  */
    neuron* neurons;/**< Array di neuroni  */
};
 /**
 * @brief Struttura della sinapsi
 */
struct sinapsi
{
    _TYPE_PRECISION weight,delta;/**< Peso a cui sono soggetti i valori passanti per una sinapsi, il delta è l'errore generato in precedenza da quella sinapsi */
    neuron *in,*out;/**< Array di neuroni in input e in output */
};
 /**
 * @brief Struttura della rete neurale brain
 */
struct brain{
    int num_inputs, num_hidden_layers, *num_neurons_hidden_layer, num_outputs;/**< Numero di neuroni in input, numero di strati nascosti, array di neuroni che rapresentano gli strati nascosti, numero di neuroni output */
    double* range,* min_max;/**< Campo di esistenza nel quale verrano fatti i calcoli, Array per rapresentare campo di esistenza degli input */
    _TYPE_PRECISION error;/**< Errore complessivo della rete neurale */
    _TYPE_PRECISION (*error_function)(neuron* n, _TYPE_PRECISION desidered);/**< Puntatore alla funzione per il calcolo del errore, SSE oppure CEE */
    layer *layer_input;/**< Strato di input */
    layer *hidden_layer;/**< Strati nascosti */
    layer *layer_output;/**< Strato di output */

} ;

void homunculus_brain_free(homunculus_brain *brain);/**< Funzione che libera la memoria del pc dalla presenza di brain */
_TYPE_PRECISION test_brain (homunculus_brain* brain,_TYPE_PRECISION**input,int num_input);/**< Funzione per test generale di brain */
void see_brain (homunculus_brain* brain);/**< Visualizzo l'intera rete e i pesi */
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);/**< Inizializzo la rete  neurale */
void run_training (homunculus_brain* brain,const char* data_set,const char* save_data,double rate, double moment, double error, int eta);/**< Sottopongo la rete ad apprendimento */
void mod_transfer_function(layer* l,trans_function function);/**< Funzione che permette di modificare la funzione di transferimento di un intero strato */
_TYPE_PRECISION transition_tan(_TYPE_PRECISION n);/**< Funzione di trasferimento che usa la tan */
_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n);/**< Funzione di trasferimento che usa la sigmoide */
int transition_linear_int(_TYPE_PRECISION n);/**< Funzione di trasferimento che prende un double e restituisce sempre un intero */
_TYPE_PRECISION transition_linear(_TYPE_PRECISION n);/**< Fnzione di trasferimento lineare */
int transition_step(_TYPE_PRECISION n);/**< Funzione di trasferimento a step */
void normalize_dataset(const char *file_name_dataset,const char *file_name_dataset_normalized, double* range);/**< Funzione per normalizzare un intero dataset in base al range di calcolo della rete*/
_TYPE_PRECISION normalize_data ( double* range,_TYPE_PRECISION n, double min, double max);/**< Funzione per normalizzare un singolo dato */
_TYPE_PRECISION inverse_normalize_data(double* range,_TYPE_PRECISION n, double min, double max);/**< Funzione inversa alla normalizzazione */
#endif // HOMUNCULUS_H
