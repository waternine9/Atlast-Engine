#include "vector_math.cuh"
using namespace Atlast::Vectors;
using namespace Atlast::LinearAlgebra::VectorMath;
namespace Atlast
{
	namespace Algorithms
	{
		__host__ __device__ void point_triangle(Vector3<float> p, Vector3<float> a, Vector3<float> b, Vector3<float> c, float& u, float& v, float& w);
		__host__ __device__ bool ray_triangle(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector3<float> v0, Vector3<float> v1, Vector3<float> v2, float& t, float& u, float& v);
		__host__ __device__ bool ray_aabb(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector2<Vector3<float>> minmax, float& t);
		__host__ __device__ bool ray_sphere(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector3<float> center, float radius, float& t);
	}
}