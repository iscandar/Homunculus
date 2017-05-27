#include "homunculus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
/***************************************************************************
                            PROTOTIPE FUCTION
****************************************************************************/
double homunculus_random();
void homunculus_brain_free(homunculus_brain *brain);
_TYPE_PRECISION calc_potential(neuron* n);
_TYPE_PRECISION* take_output(homunculus_brain *brain);
_TYPE_PRECISION transition_personal(_TYPE_PRECISION n);
_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n);
int transition_linear(_TYPE_PRECISION n);
neuron* init_neurons(int neurons);
layer* init_layers(int n_layers, int* n_neurons);
void layers_link(layer *l_1, layer *l_2);
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);
void init_inputs(homunculus_brain *brain, int* inputs);
void propagation_layers (homunculus_brain *brain);
void calc_output_error(homunculus_brain* brain, _TYPE_PRECISION *desidered_outputs);
void calc_back_propagation_error(homunculus_brain* brain);
void see_brain (homunculus_brain* brain);
double test_brain (homunculus_brain* brain,int **input);
void run_train (homunculus_brain* brain, int** input, int epoche,_TYPE_PRECISION* desiderato);
/***************************************************************************
                            UTILITY FUCTION
****************************************************************************/

double homunculus_random()
{
    double max=2;
    double min=-2;
    time_t segs;
    srand( ( unsigned int )(segs % 65536) );
    return (((double)rand()/RAND_MAX) * (max - min) + min);
}
void homunculus_brain_free(homunculus_brain *brain)
{
    int a;
    //libero gli input
    free( brain -> layer_input -> neurons);
    free(brain -> layer_input);
    //libero gli hidden
    for(a = 0; a < brain -> num_hidden_layers; a++)
    {
        free(brain -> hidden_layer[a].neurons);
    }
    free(brain -> hidden_layer);
    //libero gli output
    free(brain -> layer_output -> neurons);
    free(brain -> layer_output);
    free(brain);
}
//calcolo del potenzione del neurone
_TYPE_PRECISION calc_potential(neuron* n)
{
    _TYPE_PRECISION v_temp = 0;
    sinapsi *s_temp = NULL ;
    int i;
    int num =(n -> num_in_links)-1; //numero di sinapsi escludendo la sinapsi del bian
    for(i = 0; i < num; i++)//conto tutti i link in entrata ma mi fermo prima del bias
    {
        s_temp = n -> in_links+i;
        v_temp += (s_temp -> weight) * (s_temp -> in -> trans_value);
    }
    s_temp = n -> in_links+num;//aggiungo anche la sinapsi
    v_temp += (s_temp -> weight);//il flusso del neurone del bias è sempre 1 quindi vario il peso del bias in seguito

    return v_temp;
}
_TYPE_PRECISION* take_output(homunculus_brain *brain)
{
    int a;
    neuron* temp_n;
    layer* output = brain -> layer_output;
    _TYPE_PRECISION* val = ( _TYPE_PRECISION*) malloc (sizeof( _TYPE_PRECISION) * output -> num_neurons);
    for( a = 0; a < output -> num_neurons; a++)
    {
        temp_n = output -> neurons+a;
        val[a]= temp_n -> trans_value;
    }
    return val;
}
/***************************************************************************
                            TRANSFER FUCTIONS
****************************************************************************/

_TYPE_PRECISION transition_personal(_TYPE_PRECISION n)
{
    return (_TYPE_PRECISION) n;
}

_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n)
{
    return (_TYPE_PRECISION )(1/(1 + exp(-n)));
}

int transition_linear(_TYPE_PRECISION n)
{
    return (int) n;
}

/***************************************************************************
                            INIT FUCTIONS
****************************************************************************/
//creazione array di neuroni
neuron* init_neurons(int neurons)
{
    neuron *n = ( neuron* ) malloc ( neurons * sizeof( neuron ) );
    int i;
    for (i = 0; i < neurons; i++)
    {
        n[i].num_in_links = 0;
        n[i].num_out_links = 0;
        n[i].error = 0;
        n[i].in_links = NULL;
        n[i].out_links = NULL;
        n[i].prop_value = (_TYPE_PRECISION) 0;
        n[i].trans_value = (_TYPE_PRECISION) 0;
        //dichiarazione di quale funzione di trasferimento per il neurone voglio usare
        n[i].trans_function=transition_sigmoid;

    }
    return n;
}

//creazione degli strati
layer* init_layers(int n_layers, int* n_neurons)
{
    layer *l = ( layer* ) malloc ( n_layers * sizeof( layer ) );
    int i;
    for(i = 0 ; i < n_layers; i++)
    {
        l[i].num_neurons = n_neurons[i];
        l[i].neurons = init_neurons(n_neurons[i]);
    }
    return l;
}


void layers_link(layer *l_1, layer *l_2)
{
    int a,b;
    for( a = 0; a < l_2 -> num_neurons; a++)
    {
        //aggiungo la sinapsi del bias
        l_2 -> neurons[a].num_in_links = (l_1 -> num_neurons) + 1;
        l_2 -> neurons[a].in_links = (sinapsi*) malloc ( (l_2 -> neurons[a].num_in_links) * sizeof (sinapsi));
        for(b = 0; b < l_1 -> num_neurons; b++)
        {
            //aggiorno le sinapsi con i dati
            l_2 -> neurons[a].in_links[b].in = l_1 -> neurons + b;
            l_2 -> neurons[a].in_links[b].out = l_2 -> neurons + a;
            l_2 -> neurons[a].in_links[b].weight = homunculus_random();
            l_2 -> neurons[a].in_links[b].delta = l_2 -> neurons[a].in_links[b].weight;
            //printf("VALORE DI INIZIO SINAPSI %d PESO = %f ; DELTA = %f \n",l_2 -> neurons[a].in_links+b,l_2 -> neurons[a].in_links[b].weight,l_2 -> neurons[a].in_links[b].delta);
        }
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].in = NULL;
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].out = l_2 -> neurons + a;
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].weight = homunculus_random();
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].delta = l_2 -> neurons[a].in_links[l_1 -> num_neurons].weight;
        //printf("VALORE DI INIZIO SINAPSI %d PESO = %f ; DELTA = %f \n",l_2 -> neurons[a].in_links+l_1 -> num_neurons,l_2 -> neurons[a].in_links[l_1 -> num_neurons].weight,l_2 -> neurons[a].in_links[l_1 -> num_neurons].delta);
    }
    for(a =0; a < l_1 -> num_neurons;a++)
    {
        l_1 -> neurons[a].num_out_links = l_2 ->num_neurons;
        //printf("[NEURONE] indirizzo_n %d\n",n_1);
        l_1 -> neurons[a].out_links =(sinapsi**) malloc(l_1 -> neurons[a].num_out_links * sizeof(sinapsi*));
        for(b=0; b < l_1 -> neurons[a].num_out_links; b++)
        {
            //printf("SONO DENTRO!\n");
            //printf(" [SINAPSI]  indirizzo di in %d indirizzo di out %d\n",n_2 ->in_links[a].in,n_2 ->in_links[a].out);//indirizzo_n deve comparire a sinistra sempre costante
            l_1 -> neurons[a].out_links[b] =l_2 -> neurons[b].in_links+a;
            //printf("SALVATO [SINAPSI]  indirizzo di in %d indirizzo di out %d\n",n_1 -> out_links[b] -> in,n_1 -> out_links[b] -> out);//indirizzo_n deve comparire a sinistra sempre costante

        }
    }


}
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs)
{   int a=0;
    homunculus_brain* brain =  malloc (sizeof(homunculus_brain));
    brain -> num_inputs = inputs;
    brain -> layer_input = init_layers( 1 , &inputs);
    brain -> num_hidden_layers = hidden_layers;
    brain -> hidden_layer = init_layers(hidden_layers, hidden_neurons);
    brain -> num_neurons_hidden_layer = hidden_neurons;
    brain -> num_outputs = outputs;
    brain -> layer_output = init_layers( 1, &outputs);
    brain -> error = 0;
    layers_link(brain -> layer_input,brain -> hidden_layer);
       for( a = 1; a < (hidden_layers); a++)
        {

            layers_link(brain -> hidden_layer+(a-1), brain -> hidden_layer+(a));
        }
        layers_link(brain -> hidden_layer + (hidden_layers-1),brain -> layer_output);

    return brain;
}

//inserimento inputs
void init_inputs(homunculus_brain *brain, int* inputs)
{
    int a;
    //printf("inizio input\n");
    for(a = 0; a < brain -> num_inputs; a++)
    {
        //printf("valore input %d \n",inputs[a]);
        brain -> layer_input -> neurons[a].prop_value = inputs[a];
        brain -> layer_input -> neurons[a].trans_value = inputs[a];
    }

}
/***************************************************************************
                            PROPAGATION FUCTION
****************************************************************************/

//propagazione
void propagation_layers (homunculus_brain *brain)
{
    int a,b;
    for(a = 0; a < brain -> num_hidden_layers; a++ )
    {
        for(b = 0; b < brain -> hidden_layer[a].num_neurons ; b++)
        {
            brain -> hidden_layer[a].neurons[b].prop_value = calc_potential(brain -> hidden_layer[a].neurons+b);//attivazione
            brain -> hidden_layer[a].neurons[b].trans_value =brain -> hidden_layer[a].neurons[b].trans_function(brain -> hidden_layer[a].neurons[b].prop_value);//valore di trasferimento
        }
    }
    //calcolo del output
    for (b = 0; b < brain -> layer_output -> num_neurons; b++)
    {
        brain -> layer_output -> neurons[b].prop_value = calc_potential(brain -> layer_output -> neurons+b);//attivazione
        brain -> layer_output -> neurons[b].trans_value = brain -> layer_output -> neurons[b].trans_function(brain -> layer_output -> neurons[b].prop_value);
    }

}
/***************************************************************************
                            OUTPUT ERROR
****************************************************************************/
void calc_output_error(homunculus_brain* brain, _TYPE_PRECISION *desidered_outputs)
{
    int a;
    _TYPE_PRECISION error = 0;
    for(a = 0; a < brain -> layer_output -> num_neurons; a++)
    {
        error+=0.5 *(desidered_outputs[a] - brain -> layer_output -> neurons[a].trans_value)*(desidered_outputs[a] - brain -> layer_output -> neurons[a].trans_value);//calcolo errore di uscita
        //printf("valore di output : %f valore atteso : %f",(float) n_temp -> trans_value,(float) desidered_outputs[a]);
        brain -> layer_output -> neurons[a].error= (desidered_outputs[a] - brain -> layer_output -> neurons[a].trans_value) * brain -> layer_output -> neurons[a].trans_value * (1 -brain -> layer_output -> neurons[a].trans_value) ;
    }
     brain -> error+=error;
}
/***************************************************************************
                            BACK PROPAGATION
****************************************************************************/
//calcolo del delta per i layer hidden
void calc_back_propagation_error(homunculus_brain* brain)
{
    int a,b,c;
    _TYPE_PRECISION error;
    for(a = (brain ->num_hidden_layers-1); 0 <= a ; a--)
    {
        for(b = 0; b < brain -> hidden_layer[a].num_neurons; b++)
        {
            error = 0;
            for (c = 0; c < brain -> hidden_layer[a].neurons[b].num_out_links; c++)
            {
                error += brain -> hidden_layer[a].neurons[b].out_links[c]-> out -> error * brain -> hidden_layer[a].neurons[b].out_links[c]-> weight;
            }
            brain -> hidden_layer[a].neurons[b].error = error * brain -> hidden_layer[a].neurons[b].trans_value * (1 - brain -> hidden_layer[a].neurons[b].trans_value);
            //printf("ERROR INTERNAL NEURON HIDDEN LAYER : %f \n",error);
        }
    }
}
/***************************************************************************
                            UPDATE WEIGHT-DELTA
****************************************************************************/
void calc_weight_delta(homunculus_brain* brain)
{
    //aggiusto le sinapsi di output
    int a,b,c;
    //printf("inizio output\n");
    for(a = 0; a < brain -> layer_output -> num_neurons ; a++)
    {
        for(b = 0 ; b < brain -> layer_output -> neurons[a].num_in_links-1; b++)
        {
            brain -> layer_output -> neurons[a].in_links[b].weight+=epsilon *brain -> layer_output -> neurons[a].error * brain -> layer_output -> neurons[a].in_links[b].in -> trans_value +alfa * brain -> layer_output -> neurons[a].in_links[b].delta;
            brain -> layer_output -> neurons[a].in_links[b].delta = brain -> layer_output -> neurons[a].error * brain -> layer_output -> neurons[a].in_links[b].in -> trans_value;
            //printf("[SINAPSI %d ] CON PESO = %f ; DELTA = %f ;\n",brain -> layer_output -> neurons[a].in_links+b,(float)brain -> layer_output -> neurons[a].in_links[b].weight,(float)brain -> layer_output -> neurons[a].in_links[b].delta);
        }
    }
    //printf("inizio hiddent\n");
    for(a = brain -> num_hidden_layers -1; 0 <= a ; a--)
    {
        for(b = 0; b < brain -> hidden_layer[a].num_neurons; b++)
        {
            for(c = 0; c < brain -> hidden_layer[a].neurons[b].num_in_links-1; c++)
            {
            brain -> hidden_layer[a].neurons[b].in_links[c].weight+=epsilon *brain -> hidden_layer[a].neurons[b].error * brain -> hidden_layer[a].neurons[b].in_links[c].in -> trans_value +alfa * brain -> hidden_layer[a].neurons[b].in_links[c].delta;
            brain -> hidden_layer[a].neurons[b].in_links[c].delta = brain -> hidden_layer[a].neurons[b].error * brain -> hidden_layer[a].neurons[b].in_links[c].in -> trans_value;
            //printf("[SINAPSI %d ] CON PESO = %f ; DELTA = %f ;\n",brain -> hidden_layer[a].neurons[b].in_links+c,(float)brain -> hidden_layer[a].neurons[b].in_links[c].weight,(float)brain -> hidden_layer[a].neurons[b].in_links[c].delta);
            }
        }
    }
}
/***************************************************************************
                            DEBUG FUCTIONS
****************************************************************************/
void see_brain (homunculus_brain* brain)
{
    /*intera stampa della rete*/
    printf("--------------------[ Dati rete ]----------------------\n");
    printf("[ Dati rete ] Numero input %d \n",brain -> num_inputs);
    printf("[ Dati rete ] Numero layer nascosti %d\n", brain -> num_hidden_layers);
    printf("[ Dati rete ] Numero di output %d \n", brain -> num_outputs);
    printf("--------------------[ Dati input ]----------------------\n");
    for(int a = 0; a < brain -> num_inputs; a++)
    {

        printf("[ Neurone %d ] Indirizzo %x ; Val_trasferimento %f \n",a,brain -> layer_input -> neurons+a,brain -> layer_input -> neurons[a].trans_value);
        printf("[ Neurone %d ]              In_link\n",a);

        for(int b=0; b < brain -> layer_input -> neurons[a].num_in_links; b++)
        {

            printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> layer_input -> neurons[a].in_links+b,brain -> layer_input -> neurons[a].in_links[b].in ,brain -> layer_input -> neurons[a].in_links[b].out);
        }
        printf("[ Neurone %d ]              Out_link\n",a);
        for(int b=0; b < brain -> layer_input -> neurons[a].num_out_links; b++)
        {

             printf("[ Neurone %d ]              [Link OUT %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> layer_input -> neurons[a].out_links[b],brain -> layer_input -> neurons[a].out_links[b]->in ,brain -> layer_input -> neurons[a].out_links[b]->out);
        }
    }
    printf("--------------------[ Dati layers ]----------------------\n");
    for(int c = 0 ; c < brain -> num_hidden_layers;c++)
    {
        printf("[ Layer %d ]              Layer %d\n",c,c);
        for(int a = 0; a <  brain -> hidden_layer[c].num_neurons; a++)
        {
            printf("[ Neurone %d ] Indirizzo %x ; Val_trasferimento %f \n",a,brain -> hidden_layer[c].neurons+a,(float) brain -> hidden_layer[c].neurons[a].trans_value);

            printf("[ Neurone %d ]              In_link\n",a);

            for(int b=0; b < brain -> hidden_layer[c].neurons[a].num_in_links; b++)
            {
                printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> hidden_layer[c].neurons[a].in_links+b,brain -> hidden_layer[c].neurons[a].in_links[b].in ,brain -> hidden_layer[c].neurons[a].in_links[b].out);
            }
            printf("[ Neurone %d ]              Out_link\n",a);
            for(int b=0; b <  brain -> hidden_layer[c].neurons[a].num_out_links; b++)
            {

                printf("[ Neurone %d ]              [Link OUT %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> hidden_layer[c].neurons[a].out_links[b],brain -> hidden_layer[c].neurons[a].out_links[b]->in ,brain -> hidden_layer[c].neurons[a].out_links[b]->out);
            }
        }
    }
    printf("--------------------[ Dati Out ]----------------------\n");
    for(int a = 0; a < brain -> num_outputs; a++)
    {

        printf("[ Neurone %d ] Indirizzo %x ; Val_trasferimento %f \n",a,brain -> layer_output -> neurons+a,brain -> layer_output -> neurons[a].trans_value);
        printf("[ Neurone %d ]              In_link\n",a);

        for(int b=0; b < brain -> layer_output -> neurons[a].num_in_links; b++)
        {

            printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> layer_output -> neurons[a].in_links+b,brain -> layer_output -> neurons[a].in_links[b].in ,brain -> layer_output -> neurons[a].in_links[b].out);
        }
        printf("[ Neurone %d ]              Out_link\n",a);
        for(int b=0; b < brain -> layer_output -> neurons[a].num_out_links; b++)
        {

             printf("[ Neurone %d ]              [Link OUT %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> layer_output -> neurons[a].out_links[b],brain -> layer_output -> neurons[a].out_links[b]->in ,brain -> layer_output -> neurons[a].out_links[b]->out);
        }
    }
}

double test_brain (homunculus_brain* brain,int **input)
{
    double error_brain =0;
        for(int i =0; i<4;i++){
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            error_brain += brain -> error;
        }
        return error_brain;
}

/***************************************************************************
                            RUN FUCTIONS
****************************************************************************/
void run_train (homunculus_brain* brain, int** input, int epoche,_TYPE_PRECISION* desiderato)
{
    epoche = 0;

    do{
            printf("------------------ prova numero: %d -------------------\n",epoche);
brain -> error =0;
//printf("Errore inizio della prova = %f \n\n",(float) brain -> error);
    for(int i =0; i<4;i++){
            //see_brain(brain);
            //inserisco gli input
        init_inputs(brain,*(input+i));
        propagation_layers(brain);
        calc_output_error(brain,desiderato+i);
        calc_back_propagation_error(brain);
        calc_weight_delta(brain);
   // see_brain(brain);
    }
    epoche+=1;
    printf("\n\n*****************Errore della prova = %f *********************\n\n",(float) brain -> error);
    if(brain -> error < error_accepted)
        break;
    }while(1);

printf("-------------------inizio prova finale-----------------------\n");
_TYPE_PRECISION* risposta;
     for(int i =0; i<4;i++){
           // printf("\ntest per valore %d %d il risultato è ",*(input+i)[0],*(input+i)[1]);
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            risposta = take_output(brain);
     printf("risultato : %f\n",(float) risposta[0]);
        }
}






