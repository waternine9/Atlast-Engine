#pragma once
#include "../../Vectors/vectors.cuh"
#include "../fastalg.cuh"
#include <math.h>
namespace Atlast
{
	namespace LinearAlgebra
	{
		namespace VectorMath
		{
			__host__ __device__ float dotf(const Atlast::Vectors::Vector3<float> _X, const Atlast::Vectors::Vector3<float> _Y);
			__host__ __device__ float dotf(const Atlast::Vectors::Vector2<float> _X, const Atlast::Vectors::Vector2<float> _Y);
			__host__ __device__ float dotf(const Atlast::Vectors::Vector4<float> _X, const Atlast::Vectors::Vector4<float> _Y);
			__host__ __device__ float dot(const Atlast::Vectors::Vector3<double> _X, const Atlast::Vectors::Vector3<double> _Y);
			__host__ __device__ float dot(const Atlast::Vectors::Vector2<double> _X, const Atlast::Vectors::Vector2<double> _Y);
			__host__ __device__ float dot(const Atlast::Vectors::Vector4<double> _X, const Atlast::Vectors::Vector4<double> _Y);
			__host__ __device__ Atlast::Vectors::Vector3<float> crossf(const Atlast::Vectors::Vector3<float> _X, const Atlast::Vectors::Vector3<float> _Y);
			__host__ __device__ Atlast::Vectors::Vector3<double> cross(const Atlast::Vectors::Vector3<double> _X, const Atlast::Vectors::Vector3<double> _Y);
			__host__ __device__ Atlast::Vectors::Vector3<float> reflectf(Atlast::Vectors::Vector3<float> _X, Atlast::Vectors::Vector3<float> _NORMAL);
			__host__ __device__ Atlast::Vectors::Vector3<double> reflect(Atlast::Vectors::Vector3<double> _X, Atlast::Vectors::Vector3<double> _NORMAL);
			__host__ __device__ float lengthf(Atlast::Vectors::Vector2<float> _X);
			__host__ __device__ float lengthf(Atlast::Vectors::Vector3<float> _X);
			__host__ __device__ float lengthf(Atlast::Vectors::Vector4<float> _X);
			__host__ __device__ double lengthf(Atlast::Vectors::Vector2<double> _X);
			__host__ __device__ double lengthf(Atlast::Vectors::Vector3<double> _X);
			__host__ __device__ double lengthf(Atlast::Vectors::Vector4<double> _X);
			__host__ __device__ Atlast::Vectors::Vector2<float> normalizef(Atlast::Vectors::Vector2<float> _X);
			__host__ __device__ Atlast::Vectors::Vector3<float> normalizef(Atlast::Vectors::Vector3<float> _X);
			__host__ __device__ Atlast::Vectors::Vector4<float> normalizef(Atlast::Vectors::Vector4<float> _X);
			__host__ __device__ Atlast::Vectors::Vector2<double> normalize(Atlast::Vectors::Vector2<double> _X);
			__host__ __device__ Atlast::Vectors::Vector3<double> normalize(Atlast::Vectors::Vector3<double> _X);
			__host__ __device__ Atlast::Vectors::Vector4<double> normalize(Atlast::Vectors::Vector4<double> _X);
			__host__ __device__ Atlast::Vectors::Vector3<float> rotatef(Atlast::Vectors::Vector3<float> v, Atlast::Vectors::Vector3<float> rotate_by, Atlast::Vectors::Vector3<float> center);
			__host__ __device__ void barycentric(Atlast::Vectors::Vector3<float> _P, Atlast::Vectors::Vector3<float> a, Atlast::Vectors::Vector3<float> b, Atlast::Vectors::Vector3<float> c, float& u, float& v, float& w);
			__host__ __device__ bool line_clip_against_plane(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::Vectors::Vector3<float> p1, Atlast::Vectors::Vector3<float> p2, Atlast::Vectors::Vector3<float>& new_vertice);
			__host__ __device__ float signed_distance(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::Vectors::Vector3<float> vertex);
		}
	}
}