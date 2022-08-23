#include "nnet.h"



static branch emptyBranch = {.data = 0.0, .weight = 0.0};

static neuron emptyNeuron = {.function = LINEAR, .size = 0, .branches = &emptyBranch, .bias = 0.0, .out = 0.0};


double drand(double high, double low){
	srand(time(NULL));
	return ((double)rand()*(high-low))/(double)RAND_MAX + low;
}

branch createBranch(double weight, double data){
	branch b = {.weight = weight, .data = data};
	return b;
}

neuron *createNeuron(double bias, functions function, int size, ...){
	neuron *n = malloc(sizeof(neuron));
	n->bias = bias;
	n->function = function;
	n->size = size;
	n->branches = malloc(size*sizeof(branch));
	va_list wargs;
	va_start(wargs, size);
	for(int i = 0; i < size; ++i)
		n->branches[i] = createBranch(va_arg(wargs, double), 0.0);
	n->out = 0;
	va_end(wargs);
	return n;
}

void addBranch(neuron *n, double weight){
	branch *tmp = malloc((n->size+1)*sizeof(branch));
	memmove(tmp, n->branches, n->size*sizeof(branch));
	tmp[n->size] = createBranch(weight, 0.0);
	n->branches = tmp;
	n->size++;
}

void addBranches(neuron *n, int size, double *data){
	for(int i = 0; i < size; ++i)
		addBranch(n, data[i]);
}

void changeFunction(neuron *n, functions function){
	n->function = function;
}

int changeWeight(neuron *n, int pos, double weight){
	if(pos >= 0 && pos < n->size){
		n->branches[pos].weight = weight;
		return 0;
	}
	return -1;
}

int inputNeuron(neuron *n, int pos, double data){
	if(pos >= 0 && pos < n->size){
		n->branches[pos].data = data;
		return 0;
	}
	return -1;
}

double rawValue(neuron *n){
	double v = n->bias;
	printf("0 ");
	for(int i = 0; i < n->size; ++i){
		printf("+ %.5lf + %.5lf*%.5lf ", n->bias, n->branches[i].data, n->branches[i].weight);
		v += n->branches[i].data*n->branches[i].weight;
	}
	printf("= %.5lf\n", v);
	return v;
}

double outputNeuron(neuron *n){
	double x = rawValue(n);
	switch(n->function){
		case LINEAR:
			n->out = x;
			break;
		case SIGMOID:
			n->out = (double)(1.0/(1.0+exp(-x)));
	}
	return n->out;
}

layer createLayer(int size, ...){
	layer l;
	l.size = size;
	l.neurons = malloc(size*sizeof(neuron*));
	for(int i = 0; i < size; ++i)
		l.neurons[i] = malloc(sizeof(neuron));
	va_list nargs;
	va_start(nargs, size);
	int done = 0;
	for(int i = 0; i < size; ++i){
		neuron *a = va_arg(nargs, neuron*);
		if(a == NULL)
			done = 1;
		if(done)
			l.neurons[i] = &emptyNeuron;
		else
			l.neurons[i] = a;
	}
	va_end(nargs);
	return l;
}

net createNet(int layers, ...){
	net nt;
	nt.layers = layers;
	nt.head = NULL;
	va_list largs;
	va_start(largs, layers);
	for(int i = 0; i < layers; ++i){
		lnode *tmp = malloc(sizeof(lnode));
		tmp->layer = va_arg(largs, layer);
		tmp->next = NULL;
		if(nt.head != NULL){ 
			lnode *act = nt.head;
			while(act->next != NULL)
				act = act->next;
			act->next = tmp;
		}else{
			nt.head = tmp;
		}
	}
	va_end(largs);
	return nt;
}

double *propagateLayer(layer *l, int size, double *data){
	double *result = malloc(l->size*sizeof(double));
	for(int i = 0; i < l->size; ++i){
		if(l->neurons[i]->size < size){
			int diff = size - l->neurons[i]->size;
			printf("Creando %d mas\n", diff);
			double *weights = malloc(diff*sizeof(double));
			for(int j = 0; j < diff; ++j)
				weights[j] = drand(2, -2);
			addBranches(l->neurons[i], diff, weights);
		}
		for(int j = 0; j < size; ++j)
			inputNeuron(l->neurons[i], j, data[j]);
		result[i] = outputNeuron(l->neurons[i]);
	}
	return result;
}

double *propagate(net *nt, ...){
	if(nt->head != NULL){
		va_list ntargs;
		va_start(ntargs, nt);
		lnode *tmp = nt->head;
		for(int i = 0; i < tmp->layer.size; ++i)
			inputNeuron(tmp->layer.neurons[i], 0, va_arg(ntargs, double));
		double *data = malloc(tmp->layer.size*sizeof(double));
		for(int i = 0; i < tmp->layer.size; ++i)
			data[i] = outputNeuron(tmp->layer.neurons[i]);
		int size;
		while(tmp->next != NULL){
			size = tmp->layer.size;
			double *tmpdata = propagateLayer(&tmp->next->layer, size, data);
			data = tmpdata;
			tmp = tmp->next;
		}
		return data;
	}
	return NULL;
}
