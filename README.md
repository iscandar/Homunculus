# Homunculus (Status in progress)  
                
Homunculus is a library for the creation and management of artificial neural networks. It is a librery
written in C language. Designed to be simple to use, reliable, and easily editable
thanks to an structure more similar to a biological neural network than a mathematics model.
The Homunculus libreia has no external dependencies and everything is on a single source file.
Homunculus library allows the creation of artificial neural networks of feed-forware type with:
1. back propagation
2. different types of transfer functions (sigmoid, tanh, step)
3. Different functions for error calculaiton (SSE, CEE)
4. Implementation of the Momentum
5. Rules for updating the learning rate and momentum

# Building

Homunculus is self-contained in two files: homunculus.c and homunculus.h. To use Homunculus, simply add those two files to your project. If you dont understend how use it, see main.c that is a simple implementation.

# Features

C Ansi with no dependencies.
Contained in a single source code and header file.
All functions have notes.
Easily extendible.
Implements backpropagation training.
Implements different tipes of error calc : SSE and CEE.
Implements "Momentum updating rule".
Includes some examples.

# Installation

$ git clone https://github.com/iscandar/Homunculus
$ make

- Creation and training of a neural network
To add Homunculus to our programs simply include the header
homunculus.h
After that it will be enough to create the variable with containing our network

homunculus_brain* brain;

int hidden_neurons[1] ={8};

// here we create an array that contains the quantity  of neurons for the hidden layers

brain= brain_init(2,1,hidden_neurons,1);

// the first parameter indicates the neurons in input, the second indicates the amount of layers hidden, the third is the array containing the amount of neurons for each layer, the last parameter  contains the number of output neurons

run_training(brain,new_dataset,brain_setting,0.8,0.3,0.001,10000);

// this function takes the pointer as input to our newly created network, the name
of the date base from which to take the input, the name of the  file in which it will be save the settings once when finished
learning, the learning_rate, momentum, the acceptable error, the amount of epochs to perform.

 - At this point our network will be created and just use these commands to use it:
 
homunculus_brain* second_brain =load_setting(brain_setting);
double *temp = run_brain(brain,input);

# Status, notes, future implementation:
Further simplification of the code.
Complete translation of comments and notes in italian to english.
Implementation of a better function for dynamic updating of momentum and learning rate

 For anyone interested in participating or contributing, contact me privately
 aeontivero@gmail.com
 
 
