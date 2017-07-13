#include "homunculus.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
/***************************************************************************
                            UNIVERSAL VARIBLES
****************************************************************************/
_TYPE_PRECISION learning_rate = 0.8;//capacita di apprendimento
_TYPE_PRECISION momentum = 0.008; //considerazione dell'apprendimento passato
_TYPE_PRECISION error_accepted = 0.1;
/***************************************************************************
                            PROTOTIPE FUCTION
****************************************************************************/
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs);//create network
neuron* init_neurons(int neurons);//init of neurons
layer* init_layers(int n_layers, int* n_neurons);//init of layers
void layers_link(layer *l_1, layer *l_2);//makes layers links
void init_inputs(homunculus_brain *brain, _TYPE_PRECISION* inputs);//init input to be calculate
void propagation_layers (homunculus_brain *brain);//propagation through the layers
_TYPE_PRECISION calc_potential(neuron* n);//calculating the potential of the neuron
void calc_output_error(homunculus_brain* brain, _TYPE_PRECISION *desidered_outputs);//calculating output error compares output brain with expected result
void calc_back_propagation_error(homunculus_brain* brain);//calculating delta for hidden layers
void calc_weight_delta(homunculus_brain* brain);//calculating weight delta
_TYPE_PRECISION* run_brain (homunculus_brain* brain,_TYPE_PRECISION* input);//run the brain with an input
void run_training (homunculus_brain* brain,const char* data_set,const char* save_data,double rate, double moment, double error, int eta);//brain training
void mod_transfer_function(layer* l,_TYPE_PRECISION (*function) (_TYPE_PRECISION));//this function can mod transfer function of a layer
_TYPE_PRECISION transition_tan(_TYPE_PRECISION n);//function that use tangenth
_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n);
int transition_step(_TYPE_PRECISION n);
_TYPE_PRECISION transition_linear(_TYPE_PRECISION n);
_TYPE_PRECISION error_sse(neuron* n, _TYPE_PRECISION desidered);//sum square error
_TYPE_PRECISION error_cee(neuron* n, _TYPE_PRECISION desidered);//cross entropi error
_TYPE_PRECISION homunculus_random();//generate random nuumbers
_TYPE_PRECISION normalize_data ( double* range,_TYPE_PRECISION n, double min, double max);
void homunculus_brain_free(homunculus_brain *brain);
_TYPE_PRECISION* take_output(homunculus_brain *brain);
homunculus_brain* load_setting(const char *file_name);
_TYPE_PRECISION** load_matrix(const char *file_name,int* num_inputs,_TYPE_PRECISION*** m_inputs);
void save_homunculus(homunculus_brain * brain,const char* file_save);
_TYPE_PRECISION test_brain (homunculus_brain* brain,_TYPE_PRECISION**input,int num_input);
void test_run_brain (homunculus_brain* brain,_TYPE_PRECISION**input,int num_input);
void momentum_rate_learning(double actual_error, double past_error);
_TYPE_PRECISION inverse_normalize_data(double* range,_TYPE_PRECISION n, double min, double max);
/***************************************************************************
                            INIT FUCTIONS
****************************************************************************/

//create neural network
homunculus_brain* brain_init(int inputs, int hidden_layers, int* hidden_neurons, int outputs)
{
	if(hidden_layers < 1 || inputs < 1 || outputs < 1 || hidden_neurons == NULL)
	{
		exit(EXIT_FAILURE);
	}
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
    brain -> range=sig_range;
    layers_link(brain -> layer_input,brain -> hidden_layer);
    for( a = 1; a < (hidden_layers); a++)
        {

            layers_link(brain -> hidden_layer+(a-1), brain -> hidden_layer+(a));
        }
        layers_link(brain -> hidden_layer + (hidden_layers-1),brain -> layer_output);

    return brain;
}

//create array neurons
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

//create hidden layer
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

//create links neurons from layers
void layers_link(layer *l_1, layer *l_2)
{
    int a,b;
    for( a = 0; a < l_2 -> num_neurons; a++)
    {
        l_2 -> neurons[a].num_in_links = (l_1 -> num_neurons) + 1; //add 1 for bias
        l_2 -> neurons[a].in_links = (sinapsi*) malloc ( (l_2 -> neurons[a].num_in_links) * sizeof (sinapsi));
        for(b = 0; b < l_1 -> num_neurons; b++)
        {
            //update synaps with data
            l_2 -> neurons[a].in_links[b].in = l_1 -> neurons + b;
            l_2 -> neurons[a].in_links[b].out = l_2 -> neurons + a;
            l_2 -> neurons[a].in_links[b].weight = homunculus_random();
            l_2 -> neurons[a].in_links[b].delta = 0;
            //printf("VALORE DI INIZIO SINAPSI %d PESO = %f ; DELTA = %f \n",l_2 -> neurons[a].in_links+b,l_2 -> neurons[a].in_links[b].weight,l_2 -> neurons[a].in_links[b].delta);
        }
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].in = NULL;
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].out = l_2 -> neurons + a;
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].weight = homunculus_random();
        l_2 -> neurons[a].in_links[l_1 -> num_neurons].delta = 0;
        //printf("VALORE DI INIZIO SINAPSI %d PESO = %f ; DELTA = %f \n",l_2 -> neurons[a].in_links+l_1 -> num_neurons,l_2 -> neurons[a].in_links[l_1 -> num_neurons].weight,l_2 -> neurons[a].in_links[l_1 -> num_neurons].delta);
    }
    for(a =0; a < l_1 -> num_neurons;a++)
    {
        l_1 -> neurons[a].num_out_links = l_2 ->num_neurons;
        //printf("[NEURONE] indirizzo_n %d\n",n_1);
        l_1 -> neurons[a].out_links =(sinapsi**) malloc(l_1 -> neurons[a].num_out_links * sizeof(sinapsi*));
        for(b=0; b < l_1 -> neurons[a].num_out_links; b++)
        {
            //printf(" [SINAPSI]  indirizzo di in %d indirizzo di out %d\n",n_2 ->in_links[a].in,n_2 ->in_links[a].out);//indirizzo_n deve comparire a sinistra sempre costante
            l_1 -> neurons[a].out_links[b] =l_2 -> neurons[b].in_links+a;
        }
    }
}


//insert input in the neural network
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
//propagation through the layers
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
    //calculating output
    for (b = 0; b < brain -> layer_output -> num_neurons; b++)
    {
        brain -> layer_output -> neurons[b].prop_value = calc_potential(brain -> layer_output -> neurons+b);//attivazione
        brain -> layer_output -> neurons[b].trans_value = brain -> layer_output -> neurons[b].trans_function(brain -> layer_output -> neurons[b].prop_value);
    }

}
//calculating the potential of the neuron
_TYPE_PRECISION calc_potential(neuron* n)
{
    _TYPE_PRECISION v_temp = 0;
    sinapsi *s_temp = NULL ;
    int i;
    int num =(n -> num_in_links)-1; //Number of synapses excluding the bias synapse
    for(i = 0; i < num; i++)
    {
        s_temp = n -> in_links+i;
        v_temp += (s_temp -> weight) * (s_temp -> in -> trans_value);
    }
    s_temp = n -> in_links+num;
    v_temp += (s_temp -> weight);

    return v_temp;
}

/***************************************************************************
                            OUTPUT ERROR
****************************************************************************/
//calculating output error compares output brain with expected result
void calc_output_error(homunculus_brain* brain, _TYPE_PRECISION *desidered_outputs)
{
    int a;
    _TYPE_PRECISION error = 0;
    for(a = 0; a < brain -> layer_output -> num_neurons; a++)
    {
        error+= brain -> error_function(brain -> layer_output -> neurons+a, desidered_outputs[a]);

        // brain -> layer_output -> neurons[a].error= (desidered_outputs[a] - brain -> layer_output -> neurons[a].trans_value) * brain -> layer_output -> neurons[a].trans_value * (1 -brain -> layer_output -> neurons[a].trans_value) ;//spostato in error_sse
      }
     brain -> error+=error / brain -> layer_output -> num_neurons;
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
            brain -> layer_output -> neurons[a].in_links[b].weight+=learning_rate *brain -> layer_output -> neurons[a].error * brain -> layer_output -> neurons[a].in_links[b].in -> trans_value +momentum * brain -> layer_output -> neurons[a].in_links[b].delta;
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
            brain -> hidden_layer[a].neurons[b].in_links[c].weight+=learning_rate *brain -> hidden_layer[a].neurons[b].error * brain -> hidden_layer[a].neurons[b].in_links[c].in -> trans_value +momentum * brain -> hidden_layer[a].neurons[b].in_links[c].delta;
            brain -> hidden_layer[a].neurons[b].in_links[c].delta = brain -> hidden_layer[a].neurons[b].error * brain -> hidden_layer[a].neurons[b].in_links[c].in -> trans_value;
            //printf("[SINAPSI %d ] CON PESO = %f ; DELTA = %f ;\n",brain -> hidden_layer[a].neurons[b].in_links+c,(float)brain -> hidden_layer[a].neurons[b].in_links[c].weight,(float)brain -> hidden_layer[a].neurons[b].in_links[c].delta);
            }
        }
    }
}

/***************************************************************************
                            RUN FUCTIONS
****************************************************************************/
_TYPE_PRECISION* run_brain (homunculus_brain* brain,_TYPE_PRECISION* input)
{
    _TYPE_PRECISION* output;
    init_inputs(brain,input);
    propagation_layers(brain);
    output = take_output(brain);
    return output;
}

void run_training (homunculus_brain* brain,const char* data_set,const char* save_data,double rate, double moment, double error, int eta)
{
    FILE *fd;
    int epoche = 0;
    int i;
    _TYPE_PRECISION** inputs;
    _TYPE_PRECISION** outputs;
    _TYPE_PRECISION temp_error;
    int num_input;
    char risp;                             //variable for answer
    if(rate > 0)
    {
        learning_rate =rate;
    }
    if(moment >= 0)
    {
        momentum = moment;
    }
    if(error > 0)
    {
        error_accepted = error;
    }
    outputs=load_matrix(data_set, &num_input,&inputs);
    fd = fopen("training.log", "w+"); // a+ (create + append) option will allow appending which is useful in a log file
    if (!fd)
    {
        printf("ERROR lettura del file data_brain.data non avvenuto");
        exit(EXIT_FAILURE);
    }
    fprintf(fd,"Data learning = %f  momentum = %f \n",learning_rate,momentum);
    do{
        //printf("------------------ prova numero: %d -------------------\n",epoche);
        brain -> error =0;
        //printf(" %d\n",num_input);
        for(i =0; i< num_input;i++){
           // printf(" %lf %lf\n",inputs[i][0],inputs[i][1]);
            init_inputs(brain,*(inputs+i));
            propagation_layers(brain);
            calc_output_error(brain,*(outputs+i));
            calc_back_propagation_error(brain);
            calc_weight_delta(brain);
            //see_brain(brain);
        }

        //printf("\n\n*****************Errore della prova = %f *********************\n\n",(float) brain -> error);
        fprintf(fd,"Errore della prova %d = %f \n",epoche,(float) brain -> error);
        if(brain -> error < error_accepted && brain -> error > 0)
            break;
        if(epoche==0)
        {
            temp_error=brain -> error;
        }
        //momentum dinamico con il tempo
        //momentum_rate_learning(brain -> error,temp_error);

        //control for epoche
        if(eta > 0 && eta == epoche)
        {
            printf("Age reached, current error = %lf\n",brain -> error);
            printf("Do you want to run another age cycle? [y / n]\n(the program will halt if the accepted error is reached)\n: ");
            scanf("%c", &risp);
            if(risp=='y') break;
            else epoche = 0;
        }
        epoche+=1;
    }while(1);
    fclose(fd);

    test_run_brain(brain,inputs,num_input);
    save_homunculus(brain,save_data);
    free(inputs);
    free(outputs);
}
//momentum and learning rate rule... not perfect
void momentum_rate_learning(double actual_error, double past_error)
{
        if (momentum < 0 || momentum > 1) momentum = 0;
        if (learning_rate < 0 || learning_rate >=1) learning_rate = 0.1;
        if(past_error < actual_error)
        {
            // printf("[learning_rate] %lf\n",learning_rate);
            momentum=0;
            if(learning_rate>=0.99)learning_rate=0.01;
            else learning_rate=learning_rate + 0.01;
        }
     /*   if(past_error > actual_error)
        {
            momentum = momentum + 0.001;
            learning_rate =learning_rate + 0.01;
        }*/

}
void momentum_rate_learning_uno(int* epoche,int eta )
{
        if (eta== *epoche)
        {
        	if(learning_rate > 0.01)
        	{
        		*epoche=0;
        		learning_rate=learning_rate-0.001;
        	}
        }
}
/***************************************************************************
                            TRANSFER FUCTIONS
****************************************************************************/
//this function can mod transfer function of a layer
void mod_transfer_function(layer* l,_TYPE_PRECISION (*function) (_TYPE_PRECISION))
{
    int i;
    for(i = 0; i < l->num_neurons; i++)
    {
        l ->neurons[i].trans_function=*function;
    }
}

_TYPE_PRECISION transition_tan(_TYPE_PRECISION n)
{
	if (n < -45.0) return -1.0;
	else if (n > 45.0) return 1.0;
    return (_TYPE_PRECISION) (exp(n) - exp(-n)) / (exp(n) + exp(-n));
}

_TYPE_PRECISION transition_sigmoid(_TYPE_PRECISION n)
{
	if (n < -45.0) return 0.0;
	else if (n > 45.0) return 1.0;
    return (_TYPE_PRECISION )(1.0/(1.0 + exp(-n)));
}

int transition_step(_TYPE_PRECISION n)
{
     if (n < 0.5) return 0.0;
	 else return 1.0;
}

_TYPE_PRECISION transition_linear(_TYPE_PRECISION n)
{
    return n;
}
/***************************************************************************
                            ERROR FUCTIONS
****************************************************************************/
_TYPE_PRECISION error_sse(neuron* n, _TYPE_PRECISION desidered)//sum square error
{
     n -> error = (desidered - n -> trans_value) *  n -> trans_value * (1 - n -> trans_value) ;
    return   (0.5*((desidered - n->trans_value)*(desidered - n -> trans_value)));

}

_TYPE_PRECISION error_cee(neuron* n, _TYPE_PRECISION desidered)//cross entropi error
{
    n -> error = (desidered - n -> trans_value);
    return -(desidered * log(n -> trans_value)+(1.0 - desidered) * log( 1.0 - n -> trans_value));
}


/***************************************************************************
                            UTILITY FUCTION
****************************************************************************/

_TYPE_PRECISION homunculus_random()
{
    double range[2]={-1.5,1.5};
    return (range[0]) + ( rand() / ( RAND_MAX / ( range[1] - range[0] ) ) );
}

_TYPE_PRECISION inverse_normalize_data(double* range,_TYPE_PRECISION n, double min, double max)
{
    return (((n-range[0])/(range[1]-range[0]))*(max-min)) + min;
}

_TYPE_PRECISION normalize_data ( double* range,_TYPE_PRECISION n, double min, double max)
{
	return range[0] + (range[1] -range [0]) * ( (n - min)/(max - min));
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
    _TYPE_PRECISION weight;
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
            brain -> layer_output -> neurons[a].in_links[b].weight=weight;
        }
    }

    return brain;
}
//TODO: scrivere funzione che salvi il dataset normalizzato
_TYPE_PRECISION** load_matrix(const char *file_name,int* num_inputs,_TYPE_PRECISION*** m_inputs)
{
    FILE* fd;
    int file_found;
    _TYPE_PRECISION** outputs = NULL;
    _TYPE_PRECISION** inputs = NULL;
    int r, c_input,c_output;
    double min,max;
    int a, b;
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
    fscanf(fd,"%d",&c_input);
    fscanf(fd,"%d",&c_output);
    fscanf(fd,"%lf",&min);
    fscanf(fd,"%lf",&max);
    *num_inputs = r;
    //printf(" num_inputs = %d\n",*num_inputs);
    inputs=(_TYPE_PRECISION**) malloc ( r * sizeof(_TYPE_PRECISION));
    for(a = 0; a < r; a++ )
    {
        inputs[a]=(_TYPE_PRECISION*) malloc ( c_input * sizeof(_TYPE_PRECISION));
    }

    if(c_output > 0)
    {
        outputs=(_TYPE_PRECISION**) malloc ( r * sizeof(_TYPE_PRECISION));
        for(a = 0; a < r; a++ )
        {
            outputs[a]=(_TYPE_PRECISION*) malloc ( c_output * sizeof(_TYPE_PRECISION));
        }

    }
    while(!feof(fd))
    {
        for(a = 0; a < r; a++ )
        {
            for(b=0; b < c_input; b++ )
            {
                 fscanf(fd,"%lf",&inputs[a][b]);
               // printf(" %lf",inputs[a][b]);
            }
            for(b=0; b < c_output; b++ )
            {
                 fscanf(fd,"%lf",&outputs[a][b]);
               //  printf(" %lf",outputs[a][b]);
            }
           // printf("\n");

        }
    }
    fclose(fd);
    *m_inputs=inputs;
    return outputs;

}
void normalize_dataset(const char *file_name_dataset,const char *file_name_dataset_normalized, double* range)
{
    FILE* fd;
    FILE* save_fd;
    int file_found;
    int r, c_input,c_output;
    double min,max,temp;
    int a, b;
    do
    {
    fd = fopen(file_name_dataset,"r");
    save_fd =fopen(file_name_dataset_normalized,"w");
    if(!fd)
    {
            printf("File %s Not Found!!\n",file_name_dataset);
            file_found=1;
            exit(EXIT_FAILURE);
    }
    else
        file_found=0;

    }while(file_found);
    fscanf(fd,"%d",&r);
    fprintf(save_fd,"%d ",r);
    fscanf(fd,"%d",&c_input);
    fprintf(save_fd,"%d ",c_input);
    fscanf(fd,"%d",&c_output);
    fprintf(save_fd,"%d ",c_output);
    fscanf(fd,"%lf",&min);
    fprintf(save_fd,"%lf ",normalize_data(range,min,min,max));
    fscanf(fd,"%lf ",&max);
    fprintf(save_fd,"%lf \n",normalize_data(range,max,min,max));
    //printf(" num_inputs = %d\n",*num_inputs);

        for(a = 0; a < r; a++ )
        {
            for(b=0; b < c_input; b++ )
            {
                 fscanf(fd,"%lf",&temp);
                 fprintf(save_fd,"%lf ",normalize_data(range,temp,min,max));
               // printf(" %lf",inputs[a][b]);
            }
            for(b=0; b < c_output; b++ )
            {
                 fscanf(fd,"%lf",&temp);
                 fprintf(save_fd,"%lf ",normalize_data(range,temp,min,max));
               //  printf(" %lf",outputs[a][b]);
            }
            fprintf(save_fd,"\n");

        }
    fclose(fd);
    fclose(save_fd);
}

void save_homunculus(homunculus_brain * brain,const char* file_save)
{
    int a,b,c;
    FILE* fd;
    fd = fopen(file_save,"w");
    if (!fd)
    {
        printf("ERROR lettura del file data_brain.data non avvenuto");
        exit(EXIT_FAILURE);
    }
    //write data brain
    fprintf(fd,"%d %d\n",brain -> num_inputs,brain -> num_hidden_layers);
    for( a=0; a < brain -> num_hidden_layers; a++)
    {
        fprintf(fd,"%d\n",brain -> num_neurons_hidden_layer[a]);
    }
    fprintf(fd,"%d\n",brain -> num_outputs);
   // fprintf(fp,"[num_neurons_input] %d\n",brain ->layer_input -> num_neurons);
    for(a=0; a < brain ->layer_input -> num_neurons;a++)
    {
        neuron* n_temp = (brain ->layer_input -> neurons) + a;
        fprintf(fd,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fd,"%lf\n",s_temp -> weight);
        }
    }
    for(c=0;c < brain -> num_hidden_layers;c++)
    {
        layer* l_temp= brain -> hidden_layer + c;
        //fprintf(fp,"%d\n",l_temp -> num_neurons);
        for(a=0; a <l_temp -> num_neurons;a++)
        {
        neuron* n_temp = (l_temp -> neurons) + a;
        fprintf(fd,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fd,"%lf\n",s_temp->weight);
        }
    }
    }
   // fprintf(fp,"%d\n",brain ->layer_output -> num_neurons);
    for(a=0; a < brain ->layer_output -> num_neurons;a++)
    {
        neuron* n_temp = (brain ->layer_output -> neurons) + a;
        fprintf(fd,"%d\n",n_temp -> num_in_links);
        for(b=0;b < n_temp -> num_in_links; b++)
        {
            sinapsi* s_temp =n_temp ->in_links + b;
            fprintf(fd,"%lf\n",s_temp -> weight);
        }
    }
    fclose(fd);

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

        printf("[ Neurone %d ] Indirizzo %p ; Val_trasferimento %f \n",a,brain -> layer_input -> neurons+a,brain -> layer_input -> neurons[a].trans_value);
        printf("[ Neurone %d ]              In_link\n",a);

        for(int b=0; b < brain -> layer_input -> neurons[a].num_in_links; b++)
        {

            printf("[ Neurone %d ]              [Link IN %p] Indirizzo neurone in %p  Indirizzo neurone out %p [Peso] %lf\n",a,brain -> layer_input -> neurons[a].in_links+b,brain -> layer_input -> neurons[a].in_links[b].in ,brain -> layer_input -> neurons[a].in_links[b].out,brain -> layer_input -> neurons[a].in_links[b].weight);
        }
        printf("[ Neurone %d ]              Out_link\n",a);
        for(int b=0; b < brain -> layer_input -> neurons[a].num_out_links; b++)
        {

             printf("[ Neurone %d ]              [Link OUT %p] Indirizzo neurone in %p  Indirizzo neurone out %p\n",a,brain -> layer_input -> neurons[a].out_links[b],brain -> layer_input -> neurons[a].out_links[b]->in ,brain -> layer_input -> neurons[a].out_links[b]->out);
        }
    }
    printf("--------------------[ Dati layers ]----------------------\n");
    for(int c = 0 ; c < brain -> num_hidden_layers;c++)
    {
        printf("[ Layer %d ]              Layer %d\n",c,c);
        for(int a = 0; a <  brain -> hidden_layer[c].num_neurons; a++)
        {
            printf("[ Neurone %d ] Indirizzo %p ; Val_trasferimento %f \n",a,brain -> hidden_layer[c].neurons+a,(float) brain -> hidden_layer[c].neurons[a].trans_value);

            printf("[ Neurone %d ]              In_link\n",a);

            for(int b=0; b < brain -> hidden_layer[c].neurons[a].num_in_links; b++)
            {
                printf("[ Neurone %d ]              [Link IN %p] Indirizzo neurone in %p  Indirizzo neurone out %p [Peso] %lf\n",a,brain -> hidden_layer[c].neurons[a].in_links+b,brain -> hidden_layer[c].neurons[a].in_links[b].in ,brain -> hidden_layer[c].neurons[a].in_links[b].out,brain -> hidden_layer[c].neurons[a].in_links[b].weight);
            }
            printf("[ Neurone %d ]              Out_link\n",a);
            for(int b=0; b <  brain -> hidden_layer[c].neurons[a].num_out_links; b++)
            {

                printf("[ Neurone %d ]              [Link OUT %p] Indirizzo neurone in %p  Indirizzo neurone out %p\n",a,brain -> hidden_layer[c].neurons[a].out_links[b],brain -> hidden_layer[c].neurons[a].out_links[b]->in ,brain -> hidden_layer[c].neurons[a].out_links[b]->out);
            }
        }
    }
    printf("--------------------[ Dati Out ]----------------------\n");
    for(int a = 0; a < brain -> num_outputs; a++)
    {

        printf("[ Neurone %d ] Indirizzo %p ; Val_trasferimento %f  num in link %d\n",a,brain -> layer_output -> neurons+a,brain -> layer_output -> neurons[a].trans_value,brain -> layer_output -> neurons[a].num_in_links);
        printf("[ Neurone %d ]              In_link\n",a);

        for(int b=0; b < brain -> layer_output -> neurons[a].num_in_links; b++)
        {

            printf("[ Neurone %d ]              [Link IN %p] Indirizzo neurone in %p  Indirizzo neurone out %p [Peso] %lf\n",a,brain -> layer_output -> neurons[a].in_links+b,brain -> layer_output -> neurons[a].in_links[b].in ,brain -> layer_output -> neurons[a].in_links[b].out,brain -> layer_output -> neurons[a].in_links[b].weight);
        }
        printf("[ Neurone %d ]              Out_link\n",a);
        for(int b=0; b < brain -> layer_output -> neurons[a].num_out_links; b++)
        {

             printf("[ Neurone %d ]              [Link OUT %p] Indirizzo neurone in %p  Indirizzo neurone out %p\n",a,brain -> layer_output -> neurons[a].out_links[b],brain -> layer_output -> neurons[a].out_links[b]->in ,brain -> layer_output -> neurons[a].out_links[b]->out);
        }
    }
}

_TYPE_PRECISION test_brain (homunculus_brain* brain,_TYPE_PRECISION**input,int num_input)
{
    _TYPE_PRECISION error_brain =0;
        for(int i =0; i<num_input;i++){
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            error_brain += brain -> error;
        }
        return error_brain;
}
void test_run_brain (homunculus_brain* brain,_TYPE_PRECISION**input,int num_input)
{
    int i,a;
    printf("-------------------inizio prova finale-----------------------\n");
    _TYPE_PRECISION* risposta =NULL;
    int dim;
     for(i =0; i<num_input;i++){
            init_inputs(brain,*(input+i));
            propagation_layers(brain);
            risposta = take_output(brain);
            printf("risultato : \n");
            dim = sizeof(risposta)/(sizeof(risposta[0]));
            for( a = 0; a < dim; a++)
            {
                 printf("%f ",(float) risposta[a]);
            }
            printf("\n");

        }
}

