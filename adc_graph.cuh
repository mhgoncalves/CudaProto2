#ifndef __ADC_GRAPH_CUH__
#define __ADC_GRAPH_CUH__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include "hash_gpu.cuh"
#include "hash_cpu.cuh"


struct NodoCPU
{
	long key;
	hash_cpu<long> Edges;
	long& operator=(const long& _a) {
		this->key = _a;
		return (this->key);
	};
	~NodoCPU() {
		Edges.Clear();
	}
};

struct NodoGPU
{
	long key;
	hash_gpu<long> Edges;
	long& operator=(const long& _a) {
		this->key = _a;
		return (this->key);
	};
	~NodoGPU() {
		Edges.Clear();
	}
};


class adcGraph
{
public:
//	__host__ __device__ adcGraph();
//	__host__ __device__ ~adcGraph();


	__host__ __device__ void Host_addNode(long u);
	__host__ __device__ void Host_addEdge(long u, long v, long qtd);
	__host__ __device__ void Host_Clear();
	__host__ __device__ hash_cpu<NodoCPU>* Host_Nodes();
	__host__ __device__ void Host_to_Device();

	__host__ __device__ void Device_addNode(long u);
	__host__ __device__ void Device_addEdge(long u, long v, long qtd);
	__host__ __device__ void Device_Clear();
	__host__ __device__ hash_gpu<NodoGPU>* Device_Nodes();

	__host__ __device__ long Size();
	__host__ __device__ void Initialize(long qtdEntries);
	__host__ __device__ void LoadRandomGraph(adcGraph* pGraph, int SubGraphSize);
	__host__ __device__ void LoadDiffGraph(adcGraph* G, adcGraph* A);





private:
	long _QtdNodos;
	hash_gpu<NodoGPU> _dvcNodes;
	hash_cpu<NodoCPU> _hstNodes;
};

__host__ __device__ long adcGraph::Size()
{
	return _QtdNodos;
}


__host__ __device__ hash_cpu<NodoCPU>* adcGraph::Host_Nodes()
{
	return &_hstNodes;
}

__host__ __device__ hash_gpu<NodoGPU>* adcGraph::Device_Nodes()
{
	return &_dvcNodes;
}


__host__ __device__ void adcGraph::Host_addNode(long u)
{
	NodoCPU* pU = _hstNodes.Find(u);
	// Se necessario adiciona o Nodo u...
	if (pU->key != u && pU->Edges.Size() < 1) {
		_hstNodes.Add(u);
		pU = _hstNodes.Find(u);
	}
}

__host__ __device__ void adcGraph::Host_addEdge(long u, long v, long qtd)
{
	NodoCPU* pU = _hstNodes.Find(u);

	// Se necessario adiciona o Nodo u...
	if (pU->key != u && !pU->Edges.Prepared()) {
		_hstNodes.Add(u);
		pU = _hstNodes.Find(u);
	}

	// Se necessario inicia o vetor de Arestas do Nodo u...
	if (pU->key == u && !pU->Edges.Prepared()) {
		pU->Edges.Initialize(qtd);
	}

	// Se necessario adiciona ao Nodo u a Aresta v...
	if (pU->key == u && pU->Edges.Prepared()) {
		long* pV = pU->Edges.Find(v);
		if ((*pV)!=v)
			pU->Edges.Add(v);
	}
}

__host__ __device__ void adcGraph::Host_Clear()
{
	_hstNodes.Clear();
}

__host__ __device__ void adcGraph::Device_addNode(long u)
{
	NodoGPU* pU = _dvcNodes.Find(u);

	// Se necessario adiciona o Nodo u...
	if (pU->key != u && !pU->Edges.Prepared()) {
		_dvcNodes.Add(u);
		pU = _dvcNodes.Find(u);
	}
}

__host__ __device__ void adcGraph::Device_addEdge(long u, long v, long qtd)
{
	NodoGPU* pU = _dvcNodes.Find(u);

	// Se necessario adiciona o Nodo u...
	if (pU->key != u && !pU->Edges.Prepared()) {
		_dvcNodes.Add(u);
		pU = _dvcNodes.Find(u);
	}

	// Se necessario inicia o vetor de Arestas do Nodo u...
	if (pU->key == u && !pU->Edges.Prepared()) {
		pU->Edges.Initialize(qtd);
	}

	// Se necessario adiciona ao Nodo u a Aresta v...
	if (pU->key == u && pU->Edges.Prepared()) {
		long* pV = pU->Edges.Find(v);
		if ((*pV) != v)
			pU->Edges.Add(v);
	}
}

__host__ __device__ void adcGraph::Device_Clear()
{
	_dvcNodes.Clear();
}



__host__ __device__ void adcGraph::Initialize(long qtdNodos)
{
	_QtdNodos = qtdNodos;
	_hstNodes.Initialize(_QtdNodos);
//	_dvcNodes.Initialize(_QtdNodos);
}


////////////////////////////////////////////////////////////////////////////////
// Inicializa o grafo pegando um vértice aleatório de G.
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ void adcGraph::LoadRandomGraph(adcGraph* pGraph,int SubGraphSize)
{
	// Iniciando o grafo.
	this->Host_Clear();

	// Recuperando a posicao "aleatoria"...
	NodoCPU* Nodo = pGraph->Host_Nodes()->Rand();

	// Adicionando o Nodo recuperado à memoria do Host...
	this->Initialize(pGraph->Size());
	for (long i = 0; i < Nodo->Edges.Size(); i++) {
		this->Host_addEdge(Nodo->key, (*Nodo->Edges.Pos(i)), Nodo->Edges.Size());
	}

	// Copiando do Host para o Device...
	this->Host_to_Device();
}


__host__ __device__ void adcGraph::Host_to_Device()
{
	// Limpando e preparando a memoria do Device...
	this->Device_Clear();
	_dvcNodes.Initialize(_hstNodes.Size());

	// Copiando Nodo a Nodo...
	for (long i = 0; i < _hstNodes.Size(); i++) {
		NodoCPU* Nodo = _hstNodes.Pos(i);
		for (long j = 0; j < Nodo->Edges.Size(); j++) {
			this->Device_addEdge( Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
		}
	}
}


__host__ __device__ void adcGraph::LoadDiffGraph(adcGraph* G, adcGraph* A)
{
	// Iniciando o grafo.
	this->Host_Clear();
	hash_cpu<NodoCPU>* NodosG = G->Host_Nodes();
	NodoCPU* NodoA = A->Host_Nodes()->Pos(0);
	this->Initialize(G->Size());

	// Copiando Nodo a Nodo...
	for (long i = 0; i < NodosG->Size(); i++) {
		NodoCPU* Nodo = NodosG->Pos(i);
		if (Nodo->key != NodoA->key) {
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				this->Host_addEdge(Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
			}
		}
	}

	// Copiando do Host para o Device...
	this->Host_to_Device();
}


#endif
