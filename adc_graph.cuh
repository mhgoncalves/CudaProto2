#ifndef __ADC_GRAPH_CUH__
#define __ADC_GRAPH_CUH__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "hash_gpu.cuh"
#include "hash_cpu.cuh"
#include "book.h"


struct NodoCPU
{
	long key;
	hash_cpu<long> Edges;
	__host__ __device__ long& operator=(const long& _a) {
		if (_a == 0) {
			for (int i = 0; i < Edges.Size(); i++) {
				(*Edges.Pos(i)) = 0;
			}
		}				
		this->key = _a;
		return (this->key);
	}
	__host__ __device__ bool operator==(const long& _Right) {
		return  this->key == _Right;
	}
	__host__ __device__ ~NodoCPU() {
		Edges.Clear();
	}
};

struct NodoGPU
{
	long key;
	hash_gpu<long> Edges;
	__host__ __device__ long& operator=(const long& _a) {
		if (_a == 0) {
			for (int i = 0; i < Edges.Size(); i++) {
				(*Edges.Pos(i)) = 0;
			}
		}
		this->key = _a;
		return (this->key);
	}
	__host__ __device__ bool operator==(const long& _Right) {
		return  this->key == _Right;
	}
	__host__ __device__ ~NodoGPU() {
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
	__host__  void Device_Clear();
	__host__ __device__ hash_gpu<NodoGPU>* Device_Nodes();

	__host__ __device__ long Size();
	__host__ __device__ void Initialize(long qtdEntries);


	__device__ long GetNode(long U, int tid);


	__host__ void Host_to_Device();
	__host__ void Device_to_Host();

	__host__   void PrepareThreadsLocks(int sizeThreads);
	__device__ NodoGPU* ThreadLockNode(int tid);

	__device__ void Device_LinkGraph(adcGraph* link);
	__host__   void Host_CommitToLink();


private:
	long _QtdNodes;
	adcGraph* _LinkGraph;
	hash_gpu<NodoGPU> _LogLinkDels;
	long* _ThreadLock;
	int _sizeThreads;

	hash_gpu<NodoGPU> _dvcNodes;
	hash_cpu<NodoCPU> _hstNodes;
};

__host__ __device__ adcGraph::adcGraph()
{
	_QtdNodes    = 0;
	_LinkGraph   = 0;
	_ThreadLock  = 0;
	_sizeThreads = 0;
	_dvcNodes.Clear();
	_hstNodes.Clear();
	_LogLinkDels.Clear();
}

__host__ __device__ adcGraph::~adcGraph()
{
	_LogLinkDels.Clear();
	_dvcNodes.Clear();
	_hstNodes.Clear();
	if (_ThreadLock) {
		HANDLE_ERROR( cudaFree(_ThreadLock) );
	}
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
/*	NodoCPU* pU = _hstNodes.Find(u);
	// Se necessario adiciona o Nodo u...
	if (pU->key != u && pU->Edges.Size() < 1) {
		_hstNodes.Add(u);
		pU = _hstNodes.Find(u);
	}*/
	_hstNodes.Add(u);
}

__host__ __device__ void adcGraph::Host_addEdge(long u, long v, long qtd)
{
	NodoCPU* pU = _hstNodes.Add(u);

/*	// Se necessario adiciona o Nodo u...
	NodoCPU* pU = _hstNodes.Find(u);

	// Se necessario adiciona o Nodo u...
	if (!pU->key != u && !pU->Edges.Prepared()) {
		_hstNodes.Add(u);
		pU = _hstNodes.Find(u);
	}*/

	// Se necessario inicia o vetor de Arestas do Nodo u...
	if (pU->key == u && !pU->Edges.Prepared()) {
		pU->Edges.Initialize(qtd);
	}

	// Se necessario adiciona ao Nodo u a Aresta v...
	if (pU->key == u && pU->Edges.Prepared()) {
//		long* pV = pU->Edges.Find(v);
//		if ((*pV) != v)
			pU->Edges.Add(v);
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
		NodoGPU* NodoDel  = _LogLinkDels.Find(u);
		NodoGPU* NodoLink = _LinkGraph->Device_Nodes()->Find(u);

		// Verifica se o Nodo u já foi adicionado ao log dos excluídos...
		if (NodoDel  &&  NodoDel->key!=u &&  !NodoDel->Edges.Prepared() &&
			NodoLink && NodoLink->key==u &&  NodoLink->Edges.Prepared() ){

			// Adiciona o Nodo u ao log dos excluídos...
			NodoDel = _LogLinkDels.Add(u);
//			NodoDel = _LogLinkDels.Find(u);
			long qtd = NodoLink->Edges.Size();
			NodoDel->Edges.Initialize(qtd);

			// Adiciona as Arestas do Nodo u ao logo dos excluídos...
			for (long j = 0; j < NodoDel->Edges.Size(); j++) {
				long v = (*NodoLink->Edges.Pos(j));
				NodoDel->Edges.Add(v);
			}
		}
	}
}

__host__ __device__ void adcGraph::Device_addNode(long u)
{
/*	NodoGPU* pU = _dvcNodes.Find(u);
	// Se necessario adiciona o Nodo u...
	if (pU->key != u && !pU->Edges.Prepared()) {
		_dvcNodes.Add(u);
		pU = _dvcNodes.Find(u);
	}*/
	_dvcNodes.Add(u);
}

__host__ __device__ void adcGraph::Device_addEdge(long u, long v, long qtd)
{
	NodoGPU* pU = _dvcNodes.Add(u);

/*	// Se necessario adiciona o Nodo u...
	NodoGPU* pU = _dvcNodes.Find(u);

	// Se necessario adiciona o Nodo u...
	if (pU->key != u && !pU->Edges.Prepared()) {
		_dvcNodes.Add(u);
		pU = _dvcNodes.Find(u);
	}*/

	// Se necessario inicia o vetor de Arestas do Nodo u...
	if (pU->key == u && !pU->Edges.Prepared()) {
		pU->Edges.Initialize(qtd);
	}

	// Se necessario adiciona ao Nodo u a Aresta v...
	if (pU->key == u && pU->Edges.Prepared()) {
//		long* pV = pU->Edges.Find(v);
//		if ((*pV) != v)
			pU->Edges.Add(v);
	}
}

__host__ __device__ void adcGraph::Device_delEdge(long u, long v)
{
	// Se não possuir link com outro Grafo, então é necessário efetuar o log dos excluídos...
	if (_LinkGraph) {
		NodoGPU* NodoDel  = _LogLinkDels.Find(u);
		NodoGPU* NodoLink = _LinkGraph->Device_Nodes()->Find(u);
		if ( NodoLink && NodoLink->key==u &&  NodoLink->Edges.Prepared() ) {

			// Verifica se é necessário adicionar o Nodo u ao log dos excluídos...
			if ( NodoDel && NodoDel->key!=u && !NodoDel->Edges.Prepared() ) {
				NodoDel = _LogLinkDels.Add(u);
//				NodoDel = _LogLinkDels.Find(u);
				long qtd = NodoLink->Edges.Size();
				NodoDel->Edges.Initialize(qtd);
			}

			// Adicionanto a Aresta ao log dos excluídos...
			long* pV = NodoLink->Edges.Find(v);
			if (pV && (*pV) == v) {
				NodoDel->Edges.Add(v);
			}
		}
	}
}

__global__ void DeviceClear( NodoGPU* HashDevice, long HashSize )
{
	for (int i = 0; i<HashSize; i++) {
		if (HashDevice[i].Edges.Prepared()) {
			HashDevice[i].Edges.Clear();
		}
	}
}
__host__ void adcGraph::Device_Clear()
{
	if ( _dvcNodes.Prepared() ) {
		DeviceClear<<<1,1>>>(_dvcNodes.ExposeHash(), _dvcNodes.Size());
		cudaDeviceSynchronize();
		_dvcNodes.Clear();
	}
}

__host__ __device__ void adcGraph::Initialize(long qtdNodos)
{
	_QtdNodes = qtdNodos;
	_hstNodes.Initialize(_QtdNodes);
}


__global__ void Initialize_Edges_On_Device(hash_gpu<long>* DeviceEdgesObj, long SizeEdge, long* BridgeEdgesDevice)
{
	DeviceEdgesObj->Initialize(SizeEdge);
	long* HashEdgesDevice = DeviceEdgesObj->ExposeHash();
	memcpy(&HashEdgesDevice, &BridgeEdgesDevice, SizeEdge*sizeof(long));
}

__host__ void adcGraph::Host_to_Device()
{
	// Limpando e preparando a memoria do Device...
	this->Device_Clear();
	_dvcNodes.Initialize(_hstNodes.Size());

	long     SizeNodo   = _hstNodes.Size();
	NodoCPU* HashHost   = _hstNodes.ExposeHash();
	NodoGPU* HashDevice = _dvcNodes.ExposeHash();

	for (int i=0; i<SizeNodo; i++) {
		HANDLE_ERROR( cudaMemcpy( &HashDevice[i].key, &HashHost[i].key, sizeof(long), cudaMemcpyHostToDevice ));

		long* BridgeEdgesDevice = 0;
		long  SizeEdge = HashHost[i].Edges.Size();
		HANDLE_ERROR(cudaMalloc(&BridgeEdgesDevice, SizeEdge*sizeof(long)));

		long* HashEdgesHost     = HashHost[i].Edges.ExposeHash();

		HANDLE_ERROR( cudaMemcpy(&BridgeEdgesDevice, &HashEdgesHost, SizeEdge*sizeof(long), cudaMemcpyHostToDevice) );

		Initialize_Edges_On_Device<<<1,1>>>( &HashDevice[i].Edges, SizeEdge, BridgeEdgesDevice );
		cudaDeviceSynchronize();

		HANDLE_ERROR( cudaFree(BridgeEdgesDevice) );
	}
/*	// Copiando Nodo a Nodo...
	for (long i = 0; i < _hstNodes.Size(); i++) {
		NodoCPU* Nodo = _hstNodes.Pos(i);
		if ( Nodo && Nodo->Edges.Prepared() ) {
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				this->Device_addEdge(Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
			}
		}
	}
*/
}

__host__ void adcGraph::Device_to_Host()
{
	// Limpando e preparando a memoria do Device...
	this->Host_Clear();
	_hstNodes.Initialize(_dvcNodes.Size());

	// Copiando Nodo a Nodo...
	for (long i = 0; i < _dvcNodes.Size(); i++) {
		NodoGPU* Nodo = _dvcNodes.Pos(i);
		if (Nodo && Nodo->Edges.Prepared()) {
			for (long j = 0; j < Nodo->Edges.Size(); j++) {
				this->Host_addEdge(Nodo->key, (*Nodo->Edges.Pos(j)), Nodo->Edges.Size());
			}
		}
	}
}


__host__ void adcGraph::PrepareThreadsLocks(int sizeThreads)
{
	_sizeThreads = sizeThreads;
	HANDLE_ERROR( cudaMalloc(&_ThreadLock   , _sizeThreads*sizeof(long) ));
	HANDLE_ERROR( cudaMemset( _ThreadLock, 0, _sizeThreads*sizeof(long) ));
}

__device__ NodoGPU* adcGraph::ThreadLockNode(int tid)
{
	long      NextNodo = _ThreadLock[tid] + (_dvcNodes.Size() % _sizeThreads);
	NodoGPU*  Nodo     = _dvcNodes.Pos( NextNodo );
	return Nodo;
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
//				_LinkGraph->Device_addNode(u);
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
