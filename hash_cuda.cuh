#ifndef __HASH_CUDA_CUH__
#define __HASH_CUDA_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include "lock.h"

#define SIZE    (100*1024*1024)
#define ELEMENTS    (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES     1024

struct Entry
{
	unsigned int key;
	void         *value;
	Entry        *next;
};

struct Table
{
	size_t count;
	Entry  **entries;
	Entry  *pool;
};


class hash_cuda
{
private:
	Table _MyTable;
	__device__ __host__ size_t hash(unsigned int key, size_t count) { return key % count; };

public:
	__host__ __device__ void Clear();
	__host__ __device__ void ResetHash(int entries, int elements);
	__device__ void add_to_table(unsigned int *keys, void **values, Lock *lock);
	__host__ __device__ void copy_table_to_host(Table &hostTable);
};


__host__ __device__ void hash_cuda::Clear()
{
	HANDLE_ERROR( cudaFree(_MyTable.pool)    );
	HANDLE_ERROR( cudaFree(_MyTable.entries) );
}

__host__ __device__ void hash_cuda::ResetHash(int entries, int elements)
{
	_MyTable.count = entries;
	HANDLE_ERROR( cudaMalloc( (void**)&_MyTable.entries, entries*sizeof(Entry*) ));
	HANDLE_ERROR( cudaMemset( _MyTable.entries, 0, entries * sizeof(Entry*) ));
	HANDLE_ERROR( cudaMalloc( (void**)&_MyTable.pool, elements*sizeof(Entry) ));
}

__device__ void hash_cuda::add_to_table(unsigned int *keys, void **values, Lock *lock)
{
	int tid = 0;  // threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 0; // blockDim.x * gridDim.x;

	while (tid < ELEMENTS) {
		unsigned int key = keys[tid];
		size_t hashValue = this->hash(key, _MyTable.count);

		for ( int i=0; i<32; i++ ) {
			if ( (tid%32)==i ) {
				Entry *location = &( _MyTable.pool[tid] );

				location->key   = key;
				location->value = values[tid];
				lock[hashValue].lock();

				location->next              = _MyTable.entries[hashValue];
				_MyTable.entries[hashValue] = location;
				lock[hashValue].unlock();
			}
		}
		tid += stride;
	}
}

void hash_cuda::copy_table_to_host(Table &hostTable)
{
	hostTable.count   = _MyTable.count;
	hostTable.entries = (Entry**) calloc(_MyTable.count, sizeof(Entry*));
	hostTable.pool    = (Entry*)  malloc(ELEMENTS * sizeof(Entry));

	HANDLE_ERROR( cudaMemcpy(hostTable.entries, _MyTable.entries, _MyTable.count * sizeof(Entry*), cudaMemcpyDeviceToHost ));
	HANDLE_ERROR( cudaMemcpy(hostTable.pool   , _MyTable.pool   , ELEMENTS       * sizeof(Entry) , cudaMemcpyDeviceToHost ));

	for (int i = 0; i<_MyTable.count; i++) {
		if (hostTable.entries[i] != NULL) {
			hostTable.entries[i] = (Entry*)((size_t)hostTable.entries[i] - (size_t)_MyTable.pool + (size_t)hostTable.pool);
		}
	}
	for (int i = 0; i<ELEMENTS; i++) {
		if (hostTable.pool[i].next != NULL) {
			hostTable.pool[i].next = (Entry*)((size_t)hostTable.pool[i].next - (size_t)_MyTable.pool + (size_t)hostTable.pool);
		}
	}
}


/*
void verify_table(const Table &dev_table)
{
	Table   table;
	copy_table_to_host(dev_table, table);

	int count = 0;
	for (size_t i = 0; i<table.count; i++) {
		Entry   *current = table.entries[i];
		while (current != NULL) {
			++count;
			if (hash(current->key, table.count) != i)
				printf("%d hashed to %ld, but was located at %ld\n", current->key, hash(current->key, table.count), i);
			current = current->next;
		}
	}
	if (count != ELEMENTS)
		printf("%d elements found in hash table.  Should be %ld\n", count, ELEMENTS);
	else
		printf("All %d elements found in hash table.\n", count);

	free(table.pool);
	free(table.entries);
}


int main(void)
{
	unsigned int *buffer = (unsigned int*)big_random_block(SIZE);

	unsigned int *dev_keys;
	void         **dev_values;
	HANDLE_ERROR( cudaMalloc((void**)&dev_keys  , SIZE) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_values, SIZE) );
	HANDLE_ERROR( cudaMemcpy(dev_keys, buffer, SIZE, cudaMemcpyHostToDevice) );
	// copy the values to dev_values here
	// filled in by user of this code example

	Table table;
	initialize_table( table, HASH_ENTRIES, ELEMENTS );

	Lock    lock[HASH_ENTRIES];
	Lock    *dev_lock;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_lock, HASH_ENTRIES*sizeof(Lock) ));
	HANDLE_ERROR( cudaMemcpy( dev_lock, lock, HASH_ENTRIES*sizeof(Lock), cudaMemcpyHostToDevice ));

	cudaEvent_t start, stop;
	HANDLE_ERROR( cudaEventCreate(&start)  );
	HANDLE_ERROR( cudaEventCreate(&stop)   );
	HANDLE_ERROR( cudaEventRecord(start,0) );

	add_to_table <<<60,256>>> (dev_keys, dev_values, table, dev_lock);

	HANDLE_ERROR( cudaEventRecord(stop,0)    );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, start, stop) );
	printf("Time to hash: %3.1f ms\n", elapsedTime);

	verify_table(table);

	HANDLE_ERROR( cudaEventDestroy(start) );
	HANDLE_ERROR( cudaEventDestroy(stop)  );
	free_table(table);
	HANDLE_ERROR( cudaFree(dev_lock)   );
	HANDLE_ERROR( cudaFree(dev_keys)   );
	HANDLE_ERROR( cudaFree(dev_values) );
	free(buffer);
	return 0;
}
*/



#endif