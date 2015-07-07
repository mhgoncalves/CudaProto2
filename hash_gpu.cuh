#ifndef __HASH_GPU_CUH__
#define __HASH_GPU_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "lock.h"


template<class TPT> class hash_gpu
{
private:
	bool _Prepared;
	long _Count;
	TPT *_pEntries;

public:
	__host__ __device__ hash_gpu() {
		_Prepared = false;
		_Count = 0;
		_pEntries = 0;
	};

	__host__ __device__ long GetHash(long key);
	__host__ __device__ void Clear();
	__host__ __device__ void Initialize(long qtdEntries);
	__host__ __device__ void Add(long key);
	__host__ __device__ void Del(long key);
	__host__ __device__ TPT* Find(long key);
	__host__ __device__ long Size();
	__host__ __device__ bool Prepared();
	__host__ __device__ TPT* Pos(long pos);
};


template<class TPT> __host__ __device__ long hash_gpu<TPT>::GetHash(long key)
{
	return (long)key % _Count;
}

template<class TPT> __host__ __device__ void hash_gpu<TPT>::Clear()
{
	HANDLE_ERROR( cudaFree(_pEntries) );
	_Prepared = false;
	_Count = 0;
}

template<class TPT> __host__ __device__ void hash_gpu<TPT>::Initialize(long qtdEntries)
{
	_Prepared = true;
	_Count = qtdEntries;
	HANDLE_ERROR(cudaMalloc(&_pEntries, _Count*sizeof(TPT)));
	HANDLE_ERROR(cudaMemset(_pEntries, 0, _Count*sizeof(TPT)));
}

template<class TPT> __host__ __device__ void hash_gpu<TPT>::Del(long key)
{
	long hashValue = this->GetHash(key);
	_pEntries[hashValue] = 0;
}

template<class TPT> __host__ __device__ void hash_gpu<TPT>::Add(long key)
{
	long hashValue = this->GetHash(key);
	_pEntries[hashValue] = key;
}

template<class TPT> __host__ __device__ TPT* hash_gpu<TPT>::Find(long key)
{
	long hashValue = this->GetHash(key);
	return &_pEntries[hashValue];
}

template<class TPT> __host__ __device__ long hash_gpu<TPT>::Size()
{
	return _Count;
}

template<class TPT> __host__ __device__ bool hash_gpu<TPT>::Prepared()
{
	return _Prepared;
}

template<class TPT> __host__ __device__ TPT* hash_gpu<TPT>::Pos(long pos)
{
	return &_pEntries[pos];
}

#endif