#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "adc_graph.cuh"
#include <vector>

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


int main()
{
	adcGraph G, A, B;
	int Cn;

	// Carregando os Grafos G, A, B...
	ReadGraphFromFile("g01.txt", &G);
	A.LoadRandomGraph(&G,20);
	B.LoadDiffGraph(&G, &A);

    return 0;
}

