#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>


typedef enum{
	LINEAR,
	SIGMOID,
} functions;

typedef struct{
	double weight;
	double data;
} branch;

typedef struct{
	functions function;
	int size;
	branch *branches;
	double bias;
	double out;
} neuron;

typedef struct{
	int size;
	neuron **neurons;
} layer;

typedef struct lnode{
	layer layer;
	struct lnode *next;
} lnode;

typedef struct{
	int layers;
	lnode *head;
} net;


double drand(double high, double low);

branch createBranch(double weight, double data);

neuron *createNeuron(double bias, functions function, int size, ...);

void addBranch(neuron *n, double weight);

void addBranches(neuron *n, int size, double *data);

void changeFunction(neuron *n, functions function);

int changeWeight(neuron *n, int pos, double weight);

int inputNeuron(neuron *n, int pos, double data);

double rawValue(neuron *n);

double outputNeuron(neuron *n);

layer createLayer(int size, ...);

net createNet(int layers, ...);

double *propagateLayer(layer *l, int size, double *data);

double *propagate(net *nt, ...);
