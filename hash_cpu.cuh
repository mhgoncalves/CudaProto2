#ifndef __HASH_CPU_CUH__
#define __HASH_CPU_CUH__

#include "book.h"

template<class TPT> class hash_cpu
{
private:
	bool _Prepared;
	long _Count;
	TPT *_pEntries;

public:
	__host__ __device__ hash_cpu() {
		_Prepared = false;
		_Count = 0;
		_pEntries = 0;
	};

	__host__ __device__ long GetHash(long key);
	__host__ __device__ void Clear();
	__host__ __device__ void Initialize(long qtdEntries);
	__host__ __device__ void Add(long key);
	__host__ __device__ TPT* Find(long key);
	__host__ __device__ TPT* Rand();
	__host__ __device__ TPT* Pos(long pos);
	__host__ __device__ long Size();
	__host__ __device__ bool Prepared();
};


template<class TPT> __host__ __device__ long hash_cpu<TPT>::GetHash(long key)
{
	return (long)key % _Count;
}

template<class TPT> __host__ __device__ void hash_cpu<TPT>::Clear()
{
	free(_pEntries);
	_Prepared = false;
	_Count = 0;
}

template<class TPT> __host__ __device__ void hash_cpu<TPT>::Initialize(long qtdEntries)
{
	_Prepared = true;
	_Count    = qtdEntries;
	_pEntries = (TPT*)calloc(_Count, sizeof(TPT));
}

template<class TPT> __host__ __device__ void hash_cpu<TPT>::Add(long key)
{
	long hashValue = this->GetHash(key);
	_pEntries[hashValue] = key;
}

template<class TPT> __host__ __device__ TPT* hash_cpu<TPT>::Find(long key)
{
	long hashValue = this->GetHash(key);
	return &_pEntries[hashValue];
}

template<class TPT> __host__ __device__ TPT* hash_cpu<TPT>::Rand()
{
	long RandPos = rand() % this->Size();
	return &_pEntries[RandPos];
}

template<class TPT> __host__ __device__ TPT* hash_cpu<TPT>::Pos(long pos)
{
	return &_pEntries[pos];
}

template<class TPT> __host__ __device__ long hash_cpu<TPT>::Size()
{
	return _Count;
}

template<class TPT> __host__ __device__ bool hash_cpu<TPT>::Prepared()
{
	return _Prepared;
}

#endif