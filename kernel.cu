#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "adc_graph.cuh"

#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>


void PrintGraph(adcGraph *G, std::string stg)
{
	std::cout << "Grafo [" << stg << "] " << G->Size() << std::endl;
	for (int i = 0; i<G->Size(); i++) {
		NodoCPU* Nodo = G->Host_Nodes()->Pos(i);
		std::cout << Nodo->key;
		for (int j = 0; j<Nodo->Edges.Size(); j++) {
			std::cout << (*Nodo->Edges.Pos(j));
		}
		std::cout << std::endl;
	}
}

void ReadGraphFromFile(char const* cFilePathName, adcGraph *pGraph)
{
	std::ifstream oFile(cFilePathName);
	std::string str;
	long qtdNodos = 0;
	long Nodo;
	std::vector<long> Edges;

	// Calculando o tamanho do grafo...
	while (getline(oFile, str)) {
		qtdNodos++;
	}
	pGraph->Initialize(qtdNodos);
	oFile.clear();
	oFile.seekg(0, std::ifstream::beg);

	// Processando o Grafo...
	while (getline(oFile, str)) {
		qtdNodos++;
		if (!str.empty()) {
			std::istringstream iss(str);
			iss >> Nodo;

			if (Nodo > 0) {
				// Adicionando o Nodo ao Grafo, e processando suas Arestas...
				pGraph->Host_addNode(Nodo);
				Edges.clear();				
				long Edge;
				while (iss >> Edge) {
					if (Edge > 0) {
						Edges.push_back(Edge);
					}
				}

				// Adicionando as Arestas do Nodo ao Grafo...
				for (long i = 0; i < Edges.size(); i++) {
					pGraph->Host_addEdge(Nodo, Edges[i], Edges.size());
				}
			}
		}
	}
	oFile.close();
}

void LoadRandomGraph( adcGraph *G, adcGraph *A, int SubGraphSize )
{
	// Recuperando em G um Nodo aleatorio...
	std::vector<NodoCPU*> Nodos;
	Nodos.push_back( G->Host_Nodes()->Rand() );

	// Recuperando os nodos de G que deverão ser inseridos em A a partir de um Nodo aleatorio...
	for (long i = 0; i<SubGraphSize && Nodos.size()<SubGraphSize; i++) {
		for (long j = 0; j<Nodos[i]->Edges.Size() && Nodos.size()<SubGraphSize; j++) {
			long v = (*Nodos[i]->Edges.Pos(j));
			Nodos.push_back(G->Host_Nodes()->Find(v));
		}
	}

	// Adicionando em A os nodos aleatorios recuperados em G...
	A->Initialize( G->Size() );
	for (int i = 0; i < Nodos.size(); i++) {
		for (int j = 0; j < Nodos[i]->Edges.Size(); j++) {
			A->Host_addEdge(Nodos[i]->key, (*Nodos[i]->Edges.Pos(j)), Nodos[i]->Edges.Size());
		}
	}
}

void LoadDiffGraph( adcGraph *B, adcGraph *G, adcGraph *A )
{
	B->Initialize( G->Size() );
	for ( int i=0; i<G->Size(); i++ ) {
		NodoCPU* Nodo = G->Host_Nodes()->Pos(i);
		if (!A->Host_Nodes()->Find(Nodo->key)) {
			for ( int j=0; j<Nodo->Edges.Size(); j++ ) {
				B->Host_addEdge( Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size() );
			}
		}
	}
}

__global__ void ADC( adcGraph* A, adcGraph* B )
{
/*	adcGraph Al, Bl;
	int tid = blockIdx.x + threadIdx.x;
	long V = B->GetNode(tid);
	Bl->Device_delEges(V);
	Bl->Device_delNode(V);
	Al->Device_addNode(V);
	Al->Device_addEges(V,A);
	NodoGPU* pNodo = B->Device_Nodes()->Find(tid);
/*
			Integer u = itr.next();
			graphA.addVertex(u);
			// Tenta adicionar as arestas remanescentes ao grafo
			this.addRemainingEdges(u, remainingEdges, graphA);
			Set<Edge> adjU = graphB.getAdjEdges(u);
			if (adjU != null) {
				Iterator<Edge> itrE = adjU.iterator();
				while (itrE.hasNext()) { // Theta(E)
					Edge e = itrE.next();
					// Se existir o outro o vertice da aresta, insere em A
					if (graphA.contains(e.v())) {
						graphA.addEdge(e, true);
					} else {
						// Caso contrario, coloca na lista de remanescente
						addRemainingEdge(e.u(), e.v(), remainingEdges);
						addRemainingEdge(e.v(), e.u(), remainingEdges);
					}
				}
				graphB.removeVertex(u);
			}
			double newCN = this.calculateCN(graphA, graphB, abEdges); // Theta(V+E)
*/
}


adcGraph* getABEdges(adcGraph* G, adcGraph* A, adcGraph* B)
{
	adcGraph* tmp = new adcGraph();
	tmp->Initialize(A->Size());

	for (int i = 0; i<G->Size(); i++) {
		NodoCPU* Nodo = G->Host_Nodes()->Pos(i);
		for (int j = 0; j<Nodo->Edges.Size(); j++) {
			long u = Nodo->key;
			long v = (*Nodo->Edges.Pos(j));
			if ((A->Host_Contains(u) && B->Host_Contains(v)) || (A->Host_Contains(v) && B->Host_Contains(u))) {
				tmp->Host_addEdge(u, v, Nodo->Edges.Size());
			}
		}
	}

	return tmp;
}


__host__ __device__ double calculateCN(adcGraph* graphA, adcGraph* graphB, adcGraph* abEdges)
{
	double eAA = graphA->getEdgeCount();
	double eBB = graphB->getEdgeCount();
	double eAB = abEdges->getEdgeCount();
	double K = eAA / (eAA + eAB);
	double eA = eAA + eAB;
	double eB = eBB + eAB;
	return K - ((eA * eB) / ((eA * eA) + (eA * eB)));
}


int main()
{
	// Carregando os Grafos G, A, B...
	adcGraph G, A, B;
	ReadGraphFromFile("g01.txt", &G);
	LoadRandomGraph(&G, &A, 3);
	LoadDiffGraph(&B, &G, &A);

	// Debug...
/*	PrintGraph(&G, "G");
	PrintGraph(&A, "A");
	PrintGraph(&B, "B"); //*/

	// Cria o grafo com as arestas entre A e B
	adcGraph* abEdges = getABEdges( G, A, B );
	double CN         = calculateCN( A, B, abEdges );

	// Guarda as arestas removidas de B e que nao foram inseridas em A
	//HashMap<Integer, HashSet<Integer>> remainingEdges = new HashMap<Integer, HashSet<Integer>>();
	adcGraph remainingEdges;

	Set<Integer> verticesInB = new HashSet<Integer>(graphB.getVertices());
	Iterator<Integer> itr = verticesInB.iterator();

	// Para cada vertice em B, faz sua insercao em A e verifica se a
	// condutancia normalizada aumentou. Em caso positivo, prossegue
	while (itr.hasNext()) { // Theta(V)
		ADC <<<1,3>>>(&A,&B);

		if (newCN > CN) {
			CN = newCN;
		} else {
			A.removeVertex(u);
			B.addVertex(u);
			if (adjU != null)
				B.addEdges(adjU, true);
		}
		abEdges = getABEdges(G, A, B); // Theta(V+E)
	}

	return 0;
}

