#pragma once
#include "../../includes.cuh"
namespace Atlast
{
	namespace Vectors
	{
		template<class T>
		class Vector3
		{
		public:
			T x;
			T y;
			T z;
			__host__ __device__ Vector3(T x, T y, T z)
			{
				this->x = x;
				this->y = y;
				this->z = z;
			}
			__host__ __device__ Vector3(T x)
			{
				this->x = x;
				this->y = x;
				this->z = x;
			}
			__host__ __device__ Vector3()
			{
				// Default
			}
			
			__host__ __device__ void operator+=(const Vector3<T> _OTHER)
			{
				this->x += _OTHER.x;
				this->y += _OTHER.y;
				this->z += _OTHER.z;
			}
			__host__ __device__ void operator-=(const Vector3<T> _OTHER)
			{
				this->x -= _OTHER.x;
				this->y -= _OTHER.y;
				this->z -= _OTHER.z;
			}
			__host__ __device__ Vector3<T> operator+(const Vector3<T> _OTHER)
			{
				return Vector3<T>(this->x + _OTHER.x, this->y + _OTHER.y, this->z + _OTHER.z);
			}
			__host__ __device__ Vector3<T> operator-(const Vector3<T> _OTHER)
			{
				return Vector3<T>(this->x - _OTHER.x, this->y - _OTHER.y, this->z - _OTHER.z);
			}
			__host__ __device__ Vector3<T> operator*(const Vector3<T> _OTHER)
			{
				return Vector3<T>(this->x * _OTHER.x, this->y * _OTHER.y, this->z * _OTHER.z);
			}
			__host__ __device__ Vector3<T> operator/(const Vector3<T> _OTHER)
			{
				return Vector3<T>(this->x / _OTHER.x, this->y / _OTHER.y, this->z / _OTHER.z);
			}
			__host__ __device__ Vector3<T> operator+(const T _OTHER)
			{
				return Vector3<T>(this->x + _OTHER, this->y + _OTHER, this->z + _OTHER);
			}
			__host__ __device__ Vector3<T> operator-(const T _OTHER)
			{
				return Vector3<T>(this->x - _OTHER, this->y - _OTHER, this->z - _OTHER);
			}
			__host__ __device__ Vector3<T> operator*(const T _OTHER)
			{
				return Vector3<T>(this->x * _OTHER, this->y * _OTHER, this->z * _OTHER);
			}
			__host__ __device__ Vector3<T> operator/(const T _OTHER)
			{
				return Vector3<T>(this->x / _OTHER, this->y / _OTHER, this->z / _OTHER);
			}
		};
	}
}