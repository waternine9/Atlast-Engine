#pragma once
#include "../../includes.cuh"
namespace Atlast
{
	namespace Vectors
	{
		template<class T>
		class Vector2
		{
		public:
			T x;
			T y;
			__host__ __device__ Vector2<T>(T x, T y)
			{
				this->x = x;
				this->y = y;
			}
			__host__ __device__ Vector2<T>(T x)
			{
				this->x = x;
				this->y = x;
			}
			__host__ __device__ Vector2<T>()
			{
				// Default
			}

			__host__ __device__ void operator+=(const Vector2<T> _OTHER)
			{
				this->x += _OTHER.x;
				this->y += _OTHER.y;
			}
			__host__ __device__ void operator-=(const Vector2<T> _OTHER)
			{
				this->x -= _OTHER.x;
				this->y -= _OTHER.y;
			}
			__host__ __device__ Vector2<T> operator+(const Vector2<T> _OTHER)
			{
				return Vector2<T>(this->x + _OTHER.x, this->y + _OTHER.y);
			}
			__host__ __device__ Vector2<T> operator-(const Vector2<T> _OTHER)
			{
				return Vector2<T>(this->x - _OTHER.x, this->y - _OTHER.y);
			}
			__host__ __device__ Vector2<T> operator*(const Vector2<T> _OTHER)
			{
				return Vector2<T>(this->x * _OTHER.x, this->y * _OTHER.y);
			}
			__host__ __device__ Vector2<T> operator/(const Vector2<T> _OTHER)
			{
				return Vector2<T>(this->x / _OTHER.x, this->y / _OTHER.y);
			}
			__host__ __device__ Vector2<T> operator+(const T _OTHER)
			{
				return Vector2<T>(this->x + _OTHER, this->y + _OTHER);
			}
			__host__ __device__ Vector2<T> operator-(const T _OTHER)
			{
				return Vector2<T>(this->x - _OTHER, this->y - _OTHER);
			}
			__host__ __device__ Vector2<T> operator*(const T _OTHER)
			{
				return Vector2<T>(this->x * _OTHER, this->y * _OTHER);
			}
			__host__ __device__ Vector2<T> operator/(const T _OTHER)
			{
				return Vector2<T>(this->x / _OTHER, this->y / _OTHER);
			}
		};
	}
}