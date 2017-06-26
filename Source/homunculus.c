#include "homunculus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/***************************************************************************
                            UNIVERSAL VARIBLES
****************************************************************************/
double epsilon = 0.8;//capacita di apprendimento
double alfa = 0.4; //considerazione dell'apprendimento passato
double error_accepted = 0.001;
/***************************************************************************
                            PROTOTIPE FUCTION
****************************************************************************/
double homunculus_random();//generate a random number between [-3,3]
void homunculus_brain_free(homunculus_brain *brain);//free the memori used by the brain
_TYPE_PRECISION calc_potential(neuron* n);//computes the potencial activation of neuron
_TYPE_PRECISION* take_output(homunculus_brain *brain);//return a vector of output
_TYPE_PRECISION transition_tan(_TYPE_PRECISION n);//this function use the tangent to computes the neuron output
_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n);//this function use the sigmoide to computes the neuron output
int transition_linear(_TYPE_PRECISION n);//this function return the activation value without previus computes
double error_sse(neuron* n, _TYPE_PRECISION desidered);
double error_cee(neuron* n, _TYPE_PRECISION desidered);
neuron* init_neurons(int neurons);
layer* init_layers(int n_layers, int* n_neurons);
void layers_link(layer *l_1, layer *l_2);
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);
void init_inputs(homunculus_brain *brain, _TYPE_PRECISION* inputs);
void propagation_layers (homunculus_brain *brain);
void calc_output_error(homunculus_brain* brain, _TYPE_PRECISION *desidered_outputs);
void calc_back_propagation_error(homunculus_brain* brain);
void see_brain (homunculus_brain* brain);
double test_brain (homunculus_brain* brain,_TYPE_PRECISION**input);
void run_train (homunculus_brain* brain, _TYPE_PRECISION** input, int epoche,_TYPE_PRECISION** desiderato,const char* file_save);
_TYPE_PRECISION** load_matrix(const char *file_name);
/***************************************************************************
                            UTILITY FUCTION
****************************************************************************/


double homunculus_random()
{
    return -3+2*((float)rand()/((float)RAND_MAX/3));//mod. [-3 and 3] to extend the range of random numbers
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
                            LOAD INPUT / SAVE DATA /LOAD DATA
****************************************************************************/
homunculus_brain* load_setting(const char *file_name)
{
    FILE* fd;
    int file_found;
    int a,b,c;
    //data brain
    int num_input;
    int num_output;
    int num_hidden_layers;
    int* num_neurons_hidden_layers;
    homunculus_brain* brain;
    //data neuron
    int num_links;
    double weight;
    do
    {

    fd = fopen(file_name,"r");
    if(!fd)
    {
            printf("File %s Not Found!!\n",file_name);
            file_found=1;
    }
    else
        file_found=0;

    }while(file_found);
    fscanf(fd,"%d %d",&num_input,&num_hidden_layers);
    num_neurons_hidden_layers = (int*) malloc (num_hidden_layers * sizeof(int));
    for( a=0; a < num_hidden_layers; a++)
    {
        fscanf(fd,"%d",&num_neurons_hidden_layers[a]);
    }
    fscanf(fd,"%d",&num_output);
    //rigenerazione della rete
    brain = brain_init(num_input, num_hidden_layers, num_neurons_hidden_layers,num_output);
    //rimessa appunto dei pesi
    for(a=0; a < num_input ;a++)
    {
        fscanf(fd,"%d",&num_links);
        for(b=0;b < num_links; b++)
        {
            fscanf(fd,"%lf",&weight);
            brain -> layer_input -> neurons[a].in_links[b].weight=weight;

        }
    }
    for(c=0;c < num_hidden_layers;c++)
    {
       // fscanf(fd,"[num_neurons_hidden %d] %d\n",NULL,NULL);
        for(a=0; a <num_neurons_hidden_layers[c];a++)
        {
            fscanf(fd,"%d",&num_links);
            for(b=0;b <num_links; b++)
            {
                fscanf(fd,"%lf",&weight);
                brain -> hidden_layer[c].neurons[a].in_links[b].weight=weight;

            }
        }
    }
   // fscanf(fd,"[num_neurons_output] %d\n",NULL);
    for(a=0; a < num_output;a++)
    {
        fscanf(fd,"%d",&num_links);
        for(b=0;b <num_links; b++)
        {
            fscanf(fd,"%lf",&weight);
            printf("ultio trovato %lf\n",weight);
            brain -> layer_output -> neurons[a].in_links[b].weight=weight;
        }
    }

    return brain;
}

_TYPE_PRECISION** load_matrix(const char *file_name)
{
    FILE* fd;
    int file_found;
    int r, c;
    int a, b;
    _TYPE_PRECISION **m;
    do
    {

    fd = fopen(file_name,"r");
    if(!fd)
    {
            printf("File %s Not Found!!\n",file_name);
            file_found=1;
    }
    else
        file_found=0;

    }while(file_found);
    fscanf(fd,"%d",&r);
    fscanf(fd,"%d",&c);
    m=(_TYPE_PRECISION**) malloc ( r * sizeof(_TYPE_PRECISION));
    for(a = 0; a < r; a++ )
    {
        m[a]=(_TYPE_PRECISION*) malloc ( c * sizeof(_TYPE_PRECISION));
    }
    while(!feof(fd))
    {
        for(a = 0; a < r; a++ )
        {
            for(b=0; b < c; b++ )
            {
                 fscanf(fd,"%lf",&m[a][b]);
               // printf("%lf ",m[a][b]);
            }
          //  printf("\n");

        }
    }
    fclose(fd);
    return m;

}
//TODO da finire
void save_homunculus(homunculus_brain * brain,const char* file_save)
{
    int a,b,c;
    FILE* fp;
    fp = fopen(file_save,"w");
    if (!fp)
    {
        printf("ERROR lettura del file data_brain.data non avvenuto");
        exit(EXIT_FAILURE);
    }
    //write data brain
    fprintf(fp,"%d %d\n",brain -> num_inputs,brain -> num_hidden_layers);
    for( a=0; a < brain -> num_hidden_layers; a++)
    {
        fprintf(fp,"%d\n",brain -> num_neurons_hidden_layer[a]);
    }
    fprintf(fp,"%d\n",brain -> num_outputs);
   // fprintf(fp,"[num_neurons_input] %d\n",brain ->layer_input -> num_neurons);
    for(a=0; a < brain ->layer_input -> num_neurons;a++)
    {
        neuron* n_temp = (brain ->layer_input -> neurons) + a;
        fprintf(fp,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fp,"%lf\n",s_temp -> weight);
        }
    }
    for(c=0;c < brain -> num_hidden_layers;c++)
    {
        layer* l_temp= brain -> hidden_layer + c;
        //fprintf(fp,"%d\n",l_temp -> num_neurons);
        for(a=0; a <l_temp -> num_neurons;a++)
        {
        neuron* n_temp = (l_temp -> neurons) + a;
        fprintf(fp,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fp,"%lf\n",s_temp->weight);
        }
    }
    }
   // fprintf(fp,"%d\n",brain ->layer_output -> num_neurons);
    for(a=0; a < brain ->layer_output -> num_neurons;a++)
    {
        neuron* n_temp = (brain ->layer_output -> neurons) + a;
        fprintf(fp,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fp,"%lf\n",s_temp -> weight);
        }
    }

}



/***************************************************************************
                            TRANSFER FUCTIONS
****************************************************************************/

_TYPE_PRECISION transition_tan(_TYPE_PRECISION n)
{
    return (_TYPE_PRECISION) ((1 - exp(-n/2))/(1 + exp(-n/2)));
}

_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n)
{
    return (_TYPE_PRECISION )(1.0/(1.0 + exp(-n)));
}

int transition_linear(_TYPE_PRECISION n)
{
    return (int) n;
}
/***************************************************************************
                            ERROR FUCTIONS
****************************************************************************/
double error_sse(neuron* n, _TYPE_PRECISION desidered)//sigmoid square error
{

    double temp = (0.5 *(desidered - n->trans_value)*(desidered - n -> trans_value));
     n -> error = (desidered - n -> trans_value) *  n -> trans_value * (1 - n -> trans_value) ;
    //printf("ERROR SEE %lf -------------- %lf\n", (float)temp,(float) n-> error);
    return   temp;

}
double error_cee(neuron* n, _TYPE_PRECISION desidered)//cross entropi error
{
    n -> error = (desidered - n -> trans_value);
    return -(desidered * log(n -> trans_value)+(1.0 - desidered) * log ( 1.0 - n -> trans_value));
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
{
    int a=0;
    homunculus_brain* brain =  malloc (sizeof(homunculus_brain));
    brain -> num_inputs = inputs;
    brain -> layer_input = init_layers( 1 , &inputs);
    brain -> num_hidden_layers = hidden_layers;
    brain -> hidden_layer = init_layers(hidden_layers, hidden_neurons);
    brain -> num_neurons_hidden_layer = hidden_neurons;
    brain -> num_outputs = outputs;
    brain -> layer_output = init_layers( 1, &outputs);
    brain -> error = 0;
    brain -> error_function = error_sse;
    layers_link(brain -> layer_input,brain -> hidden_layer);
    for( a = 1; a < (hidden_layers); a++)
        {

            layers_link(brain -> hidden_layer+(a-1), brain -> hidden_layer+(a));
        }
        layers_link(brain -> hidden_layer + (hidden_layers-1),brain -> layer_output);

    return brain;
}

//inserimento inputs
void init_inputs(homunculus_brain *brain, _TYPE_PRECISION* inputs)
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
        error+= brain -> error_function(brain -> layer_output -> neurons+a, desidered_outputs[a]);
        // brain -> layer_output -> neurons[a].error= (desidered_outputs[a] - brain -> layer_output -> neurons[a].trans_value) * brain -> layer_output -> neurons[a].trans_value * (1 -brain -> layer_output -> neurons[a].trans_value) ;//spostato in error_sse
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

            printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x [Peso] %lf\n",a,brain -> layer_input -> neurons[a].in_links+b,brain -> layer_input -> neurons[a].in_links[b].in ,brain -> layer_input -> neurons[a].in_links[b].out,brain -> layer_input -> neurons[a].in_links[b].weight);
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
                printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x [Peso] %lf\n",a,brain -> hidden_layer[c].neurons[a].in_links+b,brain -> hidden_layer[c].neurons[a].in_links[b].in ,brain -> hidden_layer[c].neurons[a].in_links[b].out,brain -> hidden_layer[c].neurons[a].in_links[b].weight);
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

        printf("[ Neurone %d ] Indirizzo %x ; Val_trasferimento %f  num in link %d\n",a,brain -> layer_output -> neurons+a,brain -> layer_output -> neurons[a].trans_value,brain -> layer_output -> neurons[a].num_in_links);
        printf("[ Neurone %d ]              In_link\n",a);

        for(int b=0; b < brain -> layer_output -> neurons[a].num_in_links; b++)
        {

            printf("[ Neurone %d ]              [Link IN %x] Indirizzo neurone in %x  Indirizzo neurone out %x [Peso] %lf\n",a,brain -> layer_output -> neurons[a].in_links+b,brain -> layer_output -> neurons[a].in_links[b].in ,brain -> layer_output -> neurons[a].in_links[b].out,brain -> layer_output -> neurons[a].in_links[b].weight);
        }
        printf("[ Neurone %d ]              Out_link\n",a);
        for(int b=0; b < brain -> layer_output -> neurons[a].num_out_links; b++)
        {

             printf("[ Neurone %d ]              [Link OUT %x] Indirizzo neurone in %x  Indirizzo neurone out %x\n",a,brain -> layer_output -> neurons[a].out_links[b],brain -> layer_output -> neurons[a].out_links[b]->in ,brain -> layer_output -> neurons[a].out_links[b]->out);
        }
    }
}

double test_brain (homunculus_brain* brain,_TYPE_PRECISION**input)
{
    double error_brain =0;
        for(int i =0; i<4;i++){//TODO  il 4 va sostituito con la quantita di prove
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            error_brain += brain -> error;
        }
        return error_brain;
}

/***************************************************************************
                            RUN FUCTIONS
****************************************************************************/
//TODO RUN_BRAIN
_TYPE_PRECISION run_brain (homunculus_brain* brain,_TYPE_PRECISION**input)
{
    int num_input=4;
    int i,a;
    printf("-------------------inizio prova finale-----------------------\n");
    _TYPE_PRECISION* risposta;
     for(int i =0; i<num_input;i++){

            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            risposta = take_output(brain);
            printf("risultato : \n");
            for( a = 0; a < 1; a++)
            {
                 printf("%f ",(float) risposta[a]);
            }
            printf("\n");

        }
}
void run_train (homunculus_brain* brain, _TYPE_PRECISION** input, int epoche,_TYPE_PRECISION** desiderato,const char* file_save)
{
    FILE *f;
    f = fopen("training.log", "w+"); // a+ (create + append) option will allow appending which is useful in a log file
    if (f == NULL) { /* Something is wrong   */}
    epoche = 0;
    int num_input=4;//958; //TODO VA SETTATO IN AUTOMATICO CON LA QUANTITA DI PROVE IN INPUT
   // double temp_error = error_accepted;
    do{
        printf("------------------ prova numero: %d -------------------\n",epoche);
        // fprintf(f,"------------------ prova numero: %d -------------------\n",epoche);
        brain -> error =0;
        //printf("Errore inizio della prova = %f \n\n",(float) brain -> error);
        for(int i =0; i<num_input;i++){
            //see_brain(brain);
            //inserisco gli input
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            calc_output_error(brain,*(desiderato+i));
            calc_back_propagation_error(brain);
            calc_weight_delta(brain);
            // see_brain(brain);
        }

        printf("\n\n*****************Errore della prova = %f *********************\n\n",(float) brain -> error);
        fprintf(f,"Errore della prova %d = %f \n",epoche,(float) brain -> error);
        if(brain -> error < error_accepted)
            break;
    //epsilon dinamico con il tempo
  /* if(temp_error < brain -> error || brain -> error > error_accepted)
    {
        printf("[Epsilon] %lf\n",epsilon);
        epsilon=epsilon - 0.00125;
        if(epsilon<0)epsilon=0.9;

        temp_error = brain -> error;
    }*/
        epoche+=1;
    }while(1);
    close(f);

    printf("-------------------inizio prova finale-----------------------\n");
    _TYPE_PRECISION* risposta;
     for(int i =0; i<num_input;i++){

            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            risposta = take_output(brain);
            printf("risultato : \n");
            for(int a = 0; a < 1; a++)
            {
                 printf("%f ",(float) risposta[a]);
            }
            printf("\n");

        }
    save_homunculus(brain,file_save);
}






