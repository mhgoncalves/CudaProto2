#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "adc_graph.cuh"

#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>

#define SIZE_THREADS 8

__host__ __device__ void PrintGraph_Host(adcGraph *G, std::string stg)
{
	std::cout << "Grafo Host [" << stg << "] " << G->Size() << std::endl;
	for (int i = 0; i<G->Size(); i++) {
		NodoCPU* Nodo = G->Host_Nodes()->Pos(i);
		std::cout << Nodo->key;
		for (int j = 0; j<Nodo->Edges.Size(); j++) {
			std::cout << (*Nodo->Edges.Pos(j));
		}
		std::cout << std::endl;
	}
}

__global__ void PrintGraph_Device(adcGraph *G, std::string stg)
{
	printf("\nGrafo Device [%s] %d\n", stg, G->Size());

	for (int i = 0; i<G->Size(); i++) {
		NodoGPU* Nodo = G->Device_Nodes()->Pos(i);
		printf("%d ", Nodo->key);
		for (int j = 0; j<Nodo->Edges.Size(); j++) {
			printf("%d ", (*Nodo->Edges.Pos(j)) );
		}
		printf("\n");
	}
}

__host__ void ReadGraphFromFile(char const* cFilePathName, adcGraph *pGraph)
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

__host__ void LoadRandomGraph(adcGraph *G, adcGraph *A, int SubGraphSize)
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
			long u = Nodos[i]->key;
			long v = (*Nodos[i]->Edges.Pos(j));
			A->Host_addEdge(u, v, Nodos[i]->Edges.Size());
		}
	}
}

__host__ void LoadDiffGraph(adcGraph *B, adcGraph *G, adcGraph *A)
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

__host__ adcGraph* getABEdges_Host(adcGraph* G, adcGraph* A, adcGraph* B)
{
	adcGraph* tmp = new adcGraph();
	tmp->Initialize( G->Size() );

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

__device__ adcGraph* getABEdges_Device(adcGraph* G, adcGraph* At, adcGraph* Bt)
{
	adcGraph* tmp = new adcGraph();
	tmp->Initialize( G->Size() );

	for (int i = 0; i<G->Size(); i++) {
		NodoGPU* Nodo = G->Device_Nodes()->Pos(i);
		for (int j = 0; j<Nodo->Edges.Size(); j++) {
			long u = Nodo->key;
			long v = (*Nodo->Edges.Pos(j));
			if ((At->Device_Contains(u) && Bt->Device_Contains(v)) || (At->Device_Contains(v) && Bt->Device_Contains(u))) {
				tmp->Device_addEdge(u, v, Nodo->Edges.Size());
			}
		}
	}

	return tmp;
}

__host__ double calculateCN_Host( adcGraph* A, adcGraph* B, adcGraph* abEdges )
{
	double eAA = A->Host_getEdgeCount();
	double eBB = B->Host_getEdgeCount();
	double eAB = abEdges->Host_getEdgeCount();
	double K   = eAA / (eAA + eAB);
	double eA  = eAA + eAB;
	double eB  = eBB + eAB;

	return K - ((eA * eB) / ((eA * eA) + (eA * eB)));
}

__device__ double calculateCN_Device(adcGraph* At, adcGraph* Bt, adcGraph* abEdges)
{
	double eAA = At->Device_getEdgeCount();
	double eBB = Bt->Device_getEdgeCount();
	double eAB = abEdges->Device_getEdgeCount();
	double K = eAA / (eAA + eAB);
	double eA = eAA + eAB;
	double eB = eBB + eAB;

	return K - ((eA * eB) / ((eA * eA) + (eA * eB)));
}

__device__ void ProcessRemainingEdges_On_Device(long u, adcGraph* RemEdgesT, adcGraph* At)
{
	NodoGPU* Nodo = RemEdgesT->Device_Nodes()->Find(u);
	if (!Nodo)
		return;

	// Tentar adicionar as Arestas do Nodo ao grafo...
	for (int i=0; i<Nodo->Edges.Size(); i++) {
		long v = (*Nodo->Edges.Pos(i));

		// Se existir em A o outro Nodo da Aresta, entao insere a Aresta em A...
		if (v>0 && At->Device_Contains(v)) {
			At->Device_addEdge(u, v, Nodo->Edges.Size());
			RemEdgesT->Device_delEdge(u, v);
		}
	}
}

__global__ void ADC(adcGraph* G, adcGraph* A, adcGraph* B, adcGraph* remainingEdges, adcGraph* pAt, adcGraph* pBt, double* pCNt, adcGraph* pRemEdgesT)
{
	int tid = blockIdx.x + threadIdx.x;
	adcGraph *At = &pAt[tid];
	adcGraph *Bt = &pBt[tid];
	adcGraph *RemEdgesT = &pRemEdgesT[tid];

	// Conecta os grafos At e Bt, e seus remanescentes, da Thread com a memoria compartilhada...
	At->Device_LinkGraph(A);
	Bt->Device_LinkGraph(B);
	RemEdgesT->Device_LinkGraph(remainingEdges);

	// Recupera um Nodo em B para o processamento da Thread corrente...
	NodoGPU* Nodo = B->ThreadLockNode(tid);
	if (!Nodo) return;
	long u = Nodo->key;
	
	// Adiciona o Nodo da Thread, e tenta adicionar as Arestas remanescentes deste Nodo a At...
	At->Device_addNode( u );
	ProcessRemainingEdges_On_Device(u, RemEdgesT, At);

	// Tentar adicionar as Arestas do Nodo a At...
	for (int i = 0; i < Nodo->Edges.Size(); i++) {
		long v = (*Nodo->Edges.Pos(i));

		// Se existir em At o outro Nodo da Aresta, entao insere a Aresta em At...
		if (At->Device_Contains(v)) {
			At->Device_addEdge(u, v, Nodo->Edges.Size() );
		} else {
			// Caso contrario, coloca a Aresta na lista de remanescente...
			RemEdgesT->Device_addEdge(u, v, Nodo->Edges.Size());
		}
	}

	// Remove de Bt o Nodo recem processedaco na Thread corrente...
	Bt->Device_delNode(u);

	// Recalcula a condutância para At e Bt...
	adcGraph* abEdges = getABEdges_Device(G, At, Bt);
	pCNt[tid]         = calculateCN_Device(At, Bt, abEdges); // Theta(V+E)
	delete abEdges;
}



int main()
{
	// Carregando os Grafos G, A, B...
	adcGraph G, A, B;
	ReadGraphFromFile("g01.txt", &G);
	LoadRandomGraph(&G, &A, 3);
	LoadDiffGraph(&B, &G, &A);

	// Debug...
	PrintGraph_Host(&G, "G");
	PrintGraph_Host(&A, "A");
	PrintGraph_Host(&B, "B");

	G.Host_to_Device();
	PrintGraph_Device<<<1,1>>>(&G, "G");
	cudaDeviceSynchronize();

	A.Host_to_Device();
	PrintGraph_Device<<<1,1>>>(&A, "A");
	cudaDeviceSynchronize();

	B.Host_to_Device();
	PrintGraph_Device<<<1,1>>>(&B, "B");
	cudaDeviceSynchronize();

	// Debug...

	B.PrepareThreadsLocks(SIZE_THREADS);

	// Cria o grafo com as arestas entre A e B
	adcGraph* abEdges = getABEdges_Host(&G, &A, &B);
	double CN         = calculateCN_Host( &A, &B, abEdges );

	// Guarda as arestas removidas de B e que nao foram inseridas em A
//	HashMap<Integer, HashSet<Integer>> remainingEdges = new HashMap<Integer, HashSet<Integer>>();
	adcGraph remainingEdges;


	// Para cada vertice em B, faz sua insercao em A e verifica se a
	// condutancia normalizada aumentou. Em caso positivo, prossegue
//	Set<Integer> verticesInB = new HashSet<Integer>(graphB.getVertices());
//	Iterator<Integer> itr = verticesInB.iterator();
	bool Continuar = true;
	while (Continuar) { // Theta(V)
		adcGraph At[SIZE_THREADS], Bt[SIZE_THREADS], RemEdgesT[SIZE_THREADS];
		double  CNt[SIZE_THREADS];

		ADC<<<1,SIZE_THREADS>>>(&G, &A, &B, &remainingEdges, At, Bt, CNt, RemEdgesT);

		// Procurando por um novo CN...
		Continuar = false;
		for (int i = 0; i < SIZE_THREADS; i++) {
			if (CNt[i]>CN) {
				// Se um novo CN foi achado, então processa a copia dos novos grafos A e B...
				Continuar = true;
				At[i].Host_CommitToLink();
				Bt[i].Host_CommitToLink();
				RemEdgesT[i].Host_CommitToLink();

				delete abEdges;
				abEdges = getABEdges_Host(&G, &A, &B); // Theta(V+E)
				CN      = calculateCN_Host(&A, &B, abEdges);
			}
		}

	}
	
	return 0;
}

