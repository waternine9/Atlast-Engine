#include "vector_math.cuh"
namespace Atlast
{
	namespace Algorithms
	{
		__host__ __device__ void point_triangle(Atlast::Vectors::Vector3<float> p, Atlast::Vectors::Vector3<float> a, Atlast::Vectors::Vector3<float> b, Atlast::Vectors::Vector3<float> c, float& u, float& v, float& w);
		__host__ __device__ bool ray_triangle(Atlast::Vectors::Vector3<float> ray_origin, Atlast::Vectors::Vector3<float> ray_direction, Atlast::Vectors::Vector3<float> v0, Atlast::Vectors::Vector3<float> v1, Atlast::Vectors::Vector3<float> v2, float& t, float& u, float& v);
	}
}