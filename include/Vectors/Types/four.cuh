#pragma once
#include "../../includes.cuh"
namespace Atlast
{
	namespace Vectors
	{
		template<class T>
		class Vector4
		{
		public:
			T x;
			T y;
			T z;
			T w;

			__host__ __device__ Vector4(T x, T y, T z, T w)
			{
				this->x = x;
				this->y = y;
				this->z = z;
				this->w = w;
			}
			__host__ __device__ Vector4(T x)
			{
				this->x = x;
				this->y = x;
				this->z = x;
				this->w = x;
			}
			__host__ __device__ Vector4()
			{
				// Default
			}

			__host__ __device__ void operator+=(const Vector4<T> _OTHER)
			{
				this->x += _OTHER.x;
				this->y += _OTHER.y;
				this->z += _OTHER.z;
				this->w += _OTHER.w;
			}
			__host__ __device__ void operator-=(const Vector4<T> _OTHER)
			{
				this->x -= _OTHER.x;
				this->y -= _OTHER.y;
				this->z -= _OTHER.z;
				this->w -= _OTHER.w;
			}
			__host__ __device__ Vector4<T> operator+(const Vector4<T> _OTHER)
			{
				return Vector4<T>(this->x + _OTHER.x, this->y + _OTHER.y, this->z + _OTHER.z, this->w + _OTHER.w);
			}
			__host__ __device__ Vector4<T> operator-(const Vector4<T> _OTHER)
			{
				return Vector4<T>(this->x - _OTHER.x, this->y - _OTHER.y, this->z - _OTHER.z, this->w - _OTHER.w);
			}
			__host__ __device__ Vector4<T> operator*(const Vector4<T> _OTHER)
			{
				return Vector4<T>(this->x * _OTHER.x, this->y * _OTHER.y, this->z * _OTHER.z, this->w * _OTHER.w);
			}
			__host__ __device__ Vector4<T> operator/(const Vector4<T> _OTHER)
			{
				return Vector4<T>(this->x / _OTHER.x, this->y / _OTHER.y, this->z / _OTHER.z, this->w / _OTHER.w);
			}
			__host__ __device__ Vector4<T> operator+(const T _OTHER)
			{
				return Vector4<T>(this->x + _OTHER, this->y + _OTHER, this->z + _OTHER, this->w + _OTHER);
			}
			__host__ __device__ Vector4<T> operator-(const T _OTHER)
			{
				return Vector4<T>(this->x - _OTHER, this->y - _OTHER, this->z - _OTHER, this->w - _OTHER);
			}
			__host__ __device__ Vector4<T> operator*(const T _OTHER)
			{
				return Vector4<T>(this->x * _OTHER, this->y * _OTHER, this->z * _OTHER, this->w * _OTHER);
			}
			__host__ __device__ Vector4<T> operator/(const T _OTHER)
			{
				return Vector4<T>(this->x / _OTHER, this->y / _OTHER, this->z / _OTHER, this->w / _OTHER);
			}

		};
	}
}