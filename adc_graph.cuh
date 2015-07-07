#ifndef __ADC_GRAPH_CUH__
#define __ADC_GRAPH_CUH__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "hash_gpu.cuh"
#include "hash_cpu.cuh"


struct NodoCPU
{
	long key;
	hash_cpu<long> Edges;
	long& operator=(const long& _a) {
		if (_a == 0) {
			for (int i = 0; i < Edges.Size(); i++) {
				(*Edges.Pos(i)) = 0;
			}
		}				
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
		if (_a == 0) {
			for (int i = 0; i < Edges.Size(); i++) {
				(*Edges.Pos(i)) = 0;
			}
		}
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
	__host__ __device__ adcGraph();
	__host__ __device__ ~adcGraph();

	__host__ __device__ long Host_getEdgeCount();
	__host__ __device__ bool Host_Contains(long u);
	__host__ __device__ void Host_addNode(long u);
	__host__ __device__ void Host_addEdge(long u, long v, long qtd);
	__host__ __device__ void Host_Clear();
	__host__ __device__ hash_cpu<NodoCPU>* Host_Nodes();

	__host__ __device__ long Device_getEdgeCount();
	__host__ __device__ bool Device_Contains(long u);
	__host__ __device__ void Device_delNode(long u);
	__host__ __device__ void Device_addNode(long u);
	__host__ __device__ void Device_addEdge(long u, long v, long qtd);
	__host__ __device__ void Device_delEdge(long u, long v);
	__host__ __device__ void Device_Clear();
	__host__ __device__ hash_gpu<NodoGPU>* Device_Nodes();

	__host__ __device__ long Size();
	__host__ __device__ void Initialize(long qtdEntries);


	__device__ long GetNode(long U, int tid);


	__host__ __device__ void Host_to_Device();
	__host__ __device__ void Device_to_Host();

	__device__ void Device_LinkGraph(adcGraph* link);
	__host__   void Host_CommitToLink();


private:
	long _QtdNodes;
	adcGraph* _LinkGraph;
	hash_gpu<NodoGPU> _LogLinkDels;

	hash_gpu<NodoGPU> _dvcNodes;
	hash_cpu<NodoCPU> _hstNodes;
};

__host__ __device__ adcGraph::adcGraph()
{
	_QtdNodes  = 0;
	_LinkGraph = 0;
	_LogLinkDels.Clear();
	_dvcNodes.Clear();
	_hstNodes.Clear();
}

__host__ __device__ adcGraph::~adcGraph()
{
	_LogLinkDels.Clear();
	_dvcNodes.Clear();
	_hstNodes.Clear();
}


__device__ long adcGraph::GetNode(long U, int tid)
{
	return (*_dvcNodes.Find(U)->Edges.Find(tid));
}


__host__ __device__ long adcGraph::Size()
{
	return _QtdNodes;
}


__host__ __device__ long adcGraph::Host_getEdgeCount()
{
	long _QtdEdges = 0;
	for (long i = 0; i < _hstNodes.Size(); i++) {
		NodoCPU* Nodo = _hstNodes.Pos(i);
		if (Nodo->key > 0) {
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				if ((*Nodo->Edges.Pos(j)) > 0) {
					_QtdEdges++;
				}
			}
		}
	}
	return _QtdEdges;
}

__host__ __device__ hash_cpu<NodoCPU>* adcGraph::Host_Nodes()
{
	return &_hstNodes;
}

__host__ __device__ hash_gpu<NodoGPU>* adcGraph::Device_Nodes()
{
	return &_dvcNodes;
}


__host__ __device__ bool adcGraph::Host_Contains(long u)
{
	NodoCPU* pU = _hstNodes.Find(u);
	return (!pU && pU->key == u);;
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
		if ((*pV) != v) {
			pU->Edges.Add(v);
		}
	}
}

__host__ __device__ void adcGraph::Host_Clear()
{
	_hstNodes.Clear();
}


__host__ __device__ long adcGraph::Device_getEdgeCount()
{
	long _QtdEdges = 0;

	// Contando as Arestas do grafo corrente...
	for (long i = 0; i < _dvcNodes.Size(); i++) {
		NodoGPU* Nodo = _dvcNodes.Pos(i);
		if (Nodo->key > 0) {
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				if ((*Nodo->Edges.Pos(j)) > 0) {
					_QtdEdges++;
				}
			}
		}
	}

	// Se possuir link com outro Grafo, então é necessário mais processamento...
	if (_LinkGraph) {

		// Somando as Arestas do grafo Link...
		for (long i = 0; i < _LinkGraph->Device_Nodes()->Size(); i++) {
			NodoGPU* Nodo = _LinkGraph->Device_Nodes()->Pos(i);
			if (Nodo->key > 0) {
				for (long j = 0; j < Nodo->Edges.Size(); j++) {
					if ((*Nodo->Edges.Pos(j)) > 0) {
						_QtdEdges++;
					}
				}
			}
		}

		// Subtraíndo as Arestas eventualmente exluídas...
		for (long i = 0; i < _LogLinkDels.Size(); i++) {
			NodoGPU* Nodo = _LogLinkDels.Pos(i);
			if (Nodo->key > 0) {
				for (long j = 0; j < Nodo->Edges.Size(); j++) {
					if ((*Nodo->Edges.Pos(j)) > 0) {
						_QtdEdges--;
					}
				}
			}
		}
	}

	return _QtdEdges;
}

__host__ __device__ bool adcGraph::Device_Contains(long u)
{
	bool bRet = false;
	
	// Se possuir link com outro Grafo, então é necessário mais processamento...
	if (_LinkGraph) {

		// Primeiro, verificando se o Nodo u foi adicionado no corrente...
		NodoGPU* pU = _dvcNodes.Find(u);
		bRet = (!pU && pU->key == u);

		// Segundo, se necessário, verificando se o Nodo u foi marcado como exluído no corrente...
		if (!bRet) {
			pU = pU = _LogLinkDels.Find(u);
			bRet = (!pU && pU->key == u);

			// Terceiro, se necessário, verificando se o Nodo u está presente no Link...
			if (!bRet) {
				pU = _LinkGraph->Device_Nodes()->Find(u);
				bRet = (!pU && pU->key == u);
			} else {
				// Se o Nodo u foi achado entre os excluídos, então ele não está no grafo...
				bRet = !bRet;
			}
		}

	// Se NÃO possuir link com outro Grafo, então basta verificar na própria tabela Hash do Device...
	} else {
		NodoGPU* pU = _dvcNodes.Find(u);
		bRet = (!pU && pU->key == u);
	}

	return bRet;
}

__host__ __device__ void adcGraph::Device_delNode(long u)
{
	// Se não possuir link com outro Grafo, então é necessário efetuar o log dos excluídos...
	if (_LinkGraph) {

		// Verifica se o Nodo u já foi adicionado ao log dos excluídos...
		NodoGPU* Nodo = _LogLinkDels.Find(u);
		if (Nodo && Nodo->key != u && !Nodo->Edges.Prepared()) {

			// Adiciona o Nodo u ao log dos excluídos...
			_LogLinkDels.Add(u);
			Nodo = _LogLinkDels.Find(u);
			long qtd = _LinkGraph->Device_Nodes()->Find(u)->Edges.Size();
			Nodo->Edges.Initialize(qtd);

			// Adiciona as Arestas do Nodo u ao logo dos excluídos...
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				long v = (*_LinkGraph->Device_Nodes()->Find(u)->Edges.Pos(j));
				Nodo->Edges.Add(v);
			}
		}
	}
//	_dvcNodes.Del(u);
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

__host__ __device__ void adcGraph::Device_delEdge(long u, long v)
{
	// Se não possuir link com outro Grafo, então é necessário efetuar o log dos excluídos...
	if (_LinkGraph) {

		// Verifica se é necessário adicionar o Nodo u ao log dos excluídos...
		NodoGPU* Nodo = _LogLinkDels.Find(u);
		if (Nodo && Nodo->key != u && !Nodo->Edges.Prepared()) {
			_LogLinkDels.Add(u);
			Nodo = _LogLinkDels.Find(u);
			long qtd = _LinkGraph->Device_Nodes()->Find(u)->Edges.Size();
			Nodo->Edges.Initialize(qtd);
		}

		// Adicionanto a Aresta ao log dos excluídos...
		long* pV = _LinkGraph->Device_Nodes()->Find(u)->Edges.Find(v);
		if (pV && (*pV) == v) {
			Nodo->Edges.Add(v);
		}
	}
/*	NodoGPU* pU = _dvcNodes.Find(u);
	if ( pU && pU->key==u ) {
		long* pV = pU->Edges.Find(v);
		if ( pV && (*pV)==v ) {
			(*pV) = 0;
		}
	}*/
}

__host__ __device__ void adcGraph::Device_Clear()
{
	_dvcNodes.Clear();
}



__host__ __device__ void adcGraph::Initialize(long qtdNodos)
{
	_QtdNodes = qtdNodos;
	_hstNodes.Initialize(_QtdNodes);
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
			this->Device_addEdge(Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
		}
	}
}

__host__ __device__ void adcGraph::Device_to_Host()
{
	// Limpando e preparando a memoria do Device...
	this->Host_Clear();
	_hstNodes.Initialize(_dvcNodes.Size());

	// Copiando Nodo a Nodo...
	for (long i = 0; i < _dvcNodes.Size(); i++) {
		NodoGPU* Nodo = _dvcNodes.Pos(i);
		for (long j = 0; j < Nodo->Edges.Size(); j++) {
			this->Host_addEdge(Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
		}
	}
}


__device__ void adcGraph::Device_LinkGraph(adcGraph* link)
{
	_LinkGraph = link;
	_dvcNodes.Initialize( link->Size() );
	_LogLinkDels.Initialize( link->Size() );
}


__host__  void adcGraph::Host_CommitToLink()
{
	// Verificando se o grafo possui link... Apenas uma segurança a mais...
	if (_LinkGraph) {

		// PRIMEIRO: avaliando as EXCLUSÕES feitas no Device que devem ser replicadas para o Link...
		for (long i = 0; i < _LogLinkDels.Size(); i++) {
			NodoGPU* Nodo = _LogLinkDels.Pos(i);
			long u = Nodo->key;
			long qtdDel = 0;

			// Apagando cada Aresta de exclusão do Nodo u...
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				long v = (*Nodo->Edges.Pos(j));
				if (v > 0) {
					_LinkGraph->Device_Nodes()->Find(u)->Edges.Del(v);
					qtdDel++;
				}
			}

			// Se todas as Arestas foram excluídas, então tratava-se de uma exclusão de Nodo...
			if (qtdDel == Nodo->Edges.Size()){
				_LinkGraph->Device_Nodes()->Del(u);
			}
		}

		// SEGUNDO: avaliando as INCLUSÕES feitas no Device que devem ser replicadas para o Link...
		for (long i = 0; i < _dvcNodes.Size(); i++) {
			NodoGPU* Nodo = _dvcNodes.Pos(i);
			long u = Nodo->key;

			// Se o Link possuir o Nodo, entao temos que avaliar cada Aresta do Nodo...
			if (_LinkGraph->Device_Contains(u)) {
				for (long j = 0; j < Nodo->Edges.Size(); j++) {
					long v = (*Nodo->Edges.Pos(j));
					if (v>0 && (*_LinkGraph->Device_Nodes()->Find(u)->Edges.Find(v))<v ) {
						_LinkGraph->Device_addEdge(u, v, Nodo->Edges.Size());
					}
				}
			
			// Senão, se o Link não possuir o Nodo, entao temos adicioná-lo ao Link... Bem como todas as Arestas do Nodo...
			} else {
				_LinkGraph->Device_addNode(u);
				for (long j = 0; j < Nodo->Edges.Size(); j++) {
					long v = (*Nodo->Edges.Pos(j));
					if ( v>0 ) {
						_LinkGraph->Device_addEdge(u, v, Nodo->Edges.Size());
					}
				}
			}
		}

		// TERCEIRO: aplicando as operações feitas no Link do seu Device para o seu Host...
		_LinkGraph->Device_to_Host();
	}
}



#endif
