#ifndef __HASH_HOSTCPU_H__
#define __HASH_HOSTCPU_H__

#include <unordered_set>

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

struct HostEdge
{
	long ID;
	std::unordered_set<long> Edges;
	HostEdge(long newID) { ID = newID; };
};

bool operator==(const HostEdge& _Left, const HostEdge& _Right) {
	return (_Left.ID == _Right.ID);
};

bool operator!=(const HostEdge& _Left, const HostEdge& _Right) {
	return (_Left.ID != _Right.ID);
};
namespace std
{
	template <>	struct hash<HostEdge>
	{
		size_t operator()(const HostEdge& hostEdge) const
		{
			return hash<long>()(hostEdge.ID);
		}
	};
};

#endif