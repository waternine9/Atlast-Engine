#include "../../Vectors/vectors.cuh"
#include "../fastalg.cuh"
#include <math.h>
using namespace Atlast::Vectors;
using namespace Atlast;
namespace Atlast
{
	namespace LinearAlgebra
	{
		namespace VectorMath
		{
			__host__ __device__ float dotf(const Vector3<float> _X, const Vector3<float> _Y);
			__host__ __device__ float dotf(const Vector2<float> _X, const Vector2<float> _Y);
			__host__ __device__ float dotf(const Vector4<float> _X, const Vector4<float> _Y);
			__host__ __device__ float dot(const Vector3<double> _X, const Vector3<double> _Y);
			__host__ __device__ float dot(const Vector2<double> _X, const Vector2<double> _Y);
			__host__ __device__ float dot(const Vector4<double> _X, const Vector4<double> _Y);
			__host__ __device__ Vector3<float> crossf(const Vector3<float> _X, const Vector3<float> _Y);
			__host__ __device__ Vector3<double> cross(const Vector3<double> _X, const Vector3<double> _Y);
			__host__ __device__ Vector3<float> reflectf(Vector3<float> _X, Vector3<float> _NORMAL);
			__host__ __device__ Vector3<double> reflect(Vector3<double> _X, Vector3<double> _NORMAL); 
			__host__ __device__ float lengthf(Vector2<float> _X);
			__host__ __device__ float lengthf(Vector3<float> _X);
			__host__ __device__ float lengthf(Vector4<float> _X);
			__host__ __device__ double lengthf(Vector2<double> _X);
			__host__ __device__ double lengthf(Vector3<double> _X);
			__host__ __device__ double lengthf(Vector4<double> _X);
			__host__ __device__ Vector2<float> normalizef(Vector2<float> _X);
			__host__ __device__ Vector3<float> normalizef(Vector3<float> _X);
			__host__ __device__ Vector4<float> normalizef(Vector4<float> _X);
			__host__ __device__ Vector2<double> normalize(Vector2<double> _X);
			__host__ __device__ Vector3<double> normalize(Vector3<double> _X);
			__host__ __device__ Vector4<double> normalize(Vector4<double> _X);
			__host__ __device__ Vector3<float> rotatef(Vector3<float> v, Vector3<float> rotate_by, Vector3<float> center);
		}
	}
}