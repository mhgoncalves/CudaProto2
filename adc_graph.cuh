#ifndef __ADC_GRAPH_CUH__
#define __ADC_GRAPH_CUH__


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>

#include <map>
#include "adc_graph.cuh"
#include "hash_cuda.cuh"


#define ADC_FILE_BUFFER 1024


struct adcNode
{
	long ID;
	__host__ __device__ adcNode() { ID = -1; };
	__host__ __device__ adcNode(long newID) { ID = newID; };
};

struct adcEdge
{
	long idParent, idChild;
	__host__ __device__ adcEdge() { idParent = -1; idChild = -1; };
	__host__ __device__ adcEdge(long NodeP, long NodeC) { idParent = NodeP; idChild = NodeC; };
};


class adcGraph
{
public:
	__host__ __device__ adcGraph();
	__host__ __device__ ~adcGraph();

	__host__ __device__ const thrust::host_vector<adcNode*>* Host_Nodes() { return &_hstNodes; };
	__host__ __device__ const thrust::host_vector<adcEdge*>* Host_Edges() { return &_hstEdges; };
	__host__ __device__ void Host_AddNode(long newID);
	__host__ __device__ void Host_AddEdge(long NodeP, long NodeC);
	__host__ __device__ void Host_ToDevice();

	__host__ __device__ const thrust::device_vector<adcNode*>* Device_Nodes() { return &_dvcNodes; };
	__host__ __device__ const thrust::device_vector<adcEdge*>* Device_Edges() { return &_dvcEdges; };
	__host__ __device__ void Device_AddNode(long newID);
	__host__ __device__ void Device_AddEdge(long NodeP, long NodeC);
	__host__ __device__ void Device_ToHost();

	__host__ __device__ thrust::device_vector<adcEdge*>* Device_GetEdges(long NodeID);
	__host__ __device__ void getRandomV(adcGraph* pGraph);

private:
	thrust::device_vector<adcNode*> _dvcNodes;
	thrust::device_vector<adcEdge*> _dvcEdges;

	thrust::host_vector<adcNode*> _hstNodes;
	thrust::host_vector<adcEdge*> _hstEdges;

	hash_cuda _dvcHashNodes;
	hash_cuda _dvcHashEdges;

	__host__ __device__ void Host_Clear();
	__host__ __device__ void Device_Clear();
};

__host__ __device__ adcGraph::adcGraph()
{
	_dvcNodes.clear();
	_dvcEdges.clear();
	_hstNodes.clear();
	_hstEdges.clear();
}

__host__ __device__ adcGraph::~adcGraph()
{
	this->Host_Clear();
	this->Device_Clear();
}

__host__ __device__ void adcGraph::Host_AddNode(long newID)
{
	_hstNodes. push_back( new adcNode(newID) );
}

__host__ __device__ void adcGraph::Host_AddEdge(long NodeP, long NodeC)
{
	_hstEdges.push_back( new adcEdge(NodeP, NodeC) );
}

__host__ __device__ void adcGraph::Host_ToDevice()
{
	this->Device_Clear();
	for (long i = 0; i < _hstNodes.size(); i++) {
		this->Device_AddNode(_hstNodes[i]->ID);
	}
	for (long i = 0; i < _hstEdges.size(); i++) {
		this->Device_AddEdge(_hstEdges[i]->idParent, _hstEdges[i]->idChild);
	}
}

__host__ __device__ void adcGraph::Host_Clear()
{
	while (!_hstNodes.empty()) {
		delete _hstNodes.back();
		_hstNodes.pop_back();
	}
	while (!_hstEdges.empty()) {
		delete _hstEdges.back();
		_hstEdges.pop_back();
	}
}


__host__ __device__ void adcGraph::Device_AddNode(long newID)
{
	adcNode* pNewNode;
	cudaMalloc(&pNewNode, sizeof(adcNode));
	pNewNode->ID = newID;
	_dvcNodes.push_back(pNewNode);
}

__host__ __device__ void adcGraph::Device_AddEdge(long NodeP, long NodeC)
{
	adcEdge* pNewEdge;
	cudaMalloc(&pNewEdge, sizeof(adcEdge));
	pNewEdge->idParent = NodeP;
	pNewEdge->idChild  = NodeC;
	_dvcEdges.push_back(pNewEdge);
}

__host__ __device__ void adcGraph::Device_ToHost()
{
	this->Host_Clear();
	for (long i = 0; i < _dvcNodes.size(); i++) {
		adcNode* pNode = _dvcNodes[i];
		this->Host_AddNode(pNode->ID);
	}
	for (long i = 0; i < _dvcEdges.size(); i++) {
		adcEdge* pEdge = _dvcEdges[i];
		this->Host_AddEdge(pEdge->idParent, pEdge->idChild);
	}
}

__host__ __device__ void adcGraph::Device_Clear()
{
	while (!_dvcNodes.empty()) {
		adcNode* pDelNode = _dvcNodes.back();
		_dvcNodes.pop_back();
		cudaFree(pDelNode);
	}
	while (!_dvcEdges.empty()) {
		adcEdge* pDelEdge = _dvcEdges.back();
		_dvcEdges.pop_back();
		cudaFree(pDelEdge);
	}
}


__host__ __device__ thrust::device_vector<adcEdge*>* adcGraph::Device_GetEdges(long NodeID)
{
	thrust::device_vector<adcEdge*>* pEdges;
	cudaMalloc(&pEdges, sizeof(thrust::device_vector<adcEdge*>));
	for (long i = 0; i < _dvcEdges.size(); i++) {
		adcEdge* pEdge = _dvcEdges[i];
		if (pEdge->idParent == NodeID) {
			pEdges->push_back(pEdge);
		}
	}
	return pEdges;
}


////////////////////////////////////////////////////////////////////////////////
// Inicializa o grafo pegando um vértice aleatório de G.
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ void adcGraph::getRandomV(adcGraph* pGraph)
{
	// Iniciando o grafo.
	this->Host_Clear();
	this->Device_Clear();

	// Recuperando a posicao "aleatoria".
	long NodePos = rand() % pGraph->Device_Nodes()->size();

	// Recuperando o Vértice e suas Arestas.
	adcNode* pNode = (*pGraph->Device_Nodes())[NodePos];
	thrust::device_vector<adcEdge*>* pEdges = this->Device_GetEdges(pNode->ID);

	// Adicionando o Vértice e suas Arestas.
	this->Device_AddNode(pNode->ID);
	for (long i = 0; i < pEdges->size(); i++) {
		adcEdge* pEdge = (*pEdges)[i];
		this->Host_AddEdge(pEdge->idParent, pEdge->idChild);
	}

	// Copiando memória do Device para o Host.
	this->Device_ToHost();
	cudaFree(pEdges);
}


#endif
