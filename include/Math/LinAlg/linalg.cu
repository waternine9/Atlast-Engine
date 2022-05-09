#include "linalg.cuh"

__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Atlast::Vectors::Vector3<float> _X, const Atlast::Vectors::Vector3<float> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y + _X.z * _Y.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Atlast::Vectors::Vector2<float> _X, const Atlast::Vectors::Vector2<float> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Atlast::Vectors::Vector4<float> _X, const Atlast::Vectors::Vector4<float> _Y)
{
	return (_X.x * _Y.y + _X.y * _Y.y + _X.z * _Y.z + _X.w * _Y.w);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Atlast::Vectors::Vector3<double> _X, const Atlast::Vectors::Vector3<double> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y + _X.z * _Y.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Atlast::Vectors::Vector2<double> _X, const Atlast::Vectors::Vector2<double> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Atlast::Vectors::Vector4<double> _X, const Atlast::Vectors::Vector4<double> _Y)
{
	return (_X.x * _Y.y + _X.y * _Y.y + _X.z * _Y.z + _X.w * _Y.w);
}
__host__ __device__ Atlast::Vectors::Vector3<float> Atlast::LinearAlgebra::VectorMath::crossf(const Atlast::Vectors::Vector3<float> _X, const Atlast::Vectors::Vector3<float> _Y)
{
	return Atlast::Vectors::Vector3<float>(
		_X.y * _Y.z - _X.z * _Y.y,
		-(_X.x * _Y.z - _X.z * _Y.x),
		_X.x * _Y.y - _X.y * _Y.x
		);
}
__host__ __device__ Atlast::Vectors::Vector3<double> Atlast::LinearAlgebra::VectorMath::cross(const Atlast::Vectors::Vector3<double> _X, const Atlast::Vectors::Vector3<double> _Y)
{
	return Atlast::Vectors::Vector3<double>(
		_X.y * _Y.z - _X.z * _Y.y,
		-(_X.x * _Y.z - _X.z * _Y.x),
		_X.x * _Y.y - _X.y * _Y.x
		);
}
__host__ __device__ Atlast::Vectors::Vector3<float> Atlast::LinearAlgebra::VectorMath::reflectf(Atlast::Vectors::Vector3<float> _X, Atlast::Vectors::Vector3<float> _NORMAL)
{
	return _X - _NORMAL * dotf(_NORMAL, _X) * 2.0f;
}
__host__ __device__ Atlast::Vectors::Vector3<double> Atlast::LinearAlgebra::VectorMath::reflect(Atlast::Vectors::Vector3<double> _X, Atlast::Vectors::Vector3<double> _NORMAL)
{
	return _X - _NORMAL * dot(_NORMAL, _X) * 2.0f;
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector2<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector3<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector4<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector2<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector3<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Atlast::Vectors::Vector4<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Atlast::Vectors::Vector2<float> Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::Vectors::Vector2<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ Atlast::Vectors::Vector3<float> Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::Vectors::Vector3<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ Atlast::Vectors::Vector4<float> Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::Vectors::Vector4<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Atlast::Vectors::Vector2<double> Atlast::LinearAlgebra::VectorMath::normalize(Atlast::Vectors::Vector2<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ Atlast::Vectors::Vector3<double> Atlast::LinearAlgebra::VectorMath::normalize(Atlast::Vectors::Vector3<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ Atlast::Vectors::Vector4<double> Atlast::LinearAlgebra::VectorMath::normalize(Atlast::Vectors::Vector4<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Atlast::Vectors::Vector3<float> Atlast::LinearAlgebra::VectorMath::rotatef(Atlast::Vectors::Vector3<float> v, Atlast::Vectors::Vector3<float> rotate_by, Atlast::Vectors::Vector3<float> center)
{
	v = v - center;
	float input1[1][3];
	float input2[3][3];
	if (rotate_by.z != 0)
	{
		float SINF = sinf(rotate_by.z);
		float COSF = cosf(rotate_by.z);
		float output2[1][3] = { { 0, 0, 0 } };
		
		input2[0][0] = COSF;
		input2[0][1] = -SINF;
		input2[0][2] = 0;
		input2[1][0] = SINF;
		input2[1][1] = COSF;
		input2[1][2] = 0;
		input2[2][0] = 0;
		input2[2][1] = 0;
		input2[2][2] = 1;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output2[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = Atlast::Vectors::Vector3<float>((float)output2[0][0], (float)output2[0][1], (float)output2[0][2]);
	}
	if (rotate_by.y != 0)
	{
		float SINF = sinf(rotate_by.y);
		float COSF = cosf(rotate_by.y);
		float output[1][3] = { { 0, 0, 0 } };
		input2[0][0] = COSF;
		input2[0][1] = 0;
		input2[0][2] = SINF;
		input2[1][0] = 0;
		input2[1][1] = 1;
		input2[1][2] = 0;
		input2[2][0] = -SINF;
		input2[2][1] = 0;
		input2[2][2] = COSF;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = Atlast::Vectors::Vector3<float>((float)output[0][0], (float)output[0][1], (float)output[0][2]);
	}
	if (rotate_by.x != 0)
	{
		float SINF = sinf(rotate_by.x);
		float COSF = cosf(rotate_by.x);
		float output4[1][3] = { { 0, 0, 0 } };
		input2[0][0] = 1;
		input2[0][1] = 0;
		input2[0][2] = 0;
		input2[1][0] = 0;
		input2[1][1] = COSF;
		input2[1][2] = -SINF;
		input2[2][0] = 0;
		input2[2][1] = SINF;
		input2[2][2] = COSF;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output4[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = Atlast::Vectors::Vector3<float>((float)output4[0][0], (float)output4[0][1], (float)output4[0][2]);
	}
	v = v + center;
	return v;
}

__host__ __device__ void Atlast::LinearAlgebra::VectorMath::barycentric(Atlast::Vectors::Vector3<float> _P, Atlast::Vectors::Vector3<float> a, Atlast::Vectors::Vector3<float> b, Atlast::Vectors::Vector3<float> c, float& u, float& v, float& w)
{
	Atlast::Vectors::Vector3<float> v0 = b - a, v1 = c - a, v2 = _P - a;
	float d00 = dotf(v0, v0);
	float d01 = dotf(v0, v1);
	float d11 = dotf(v1, v1);

	float d20 = dotf(v2, v0);
	float d21 = dotf(v2, v1);

	float denom = 1.0 / (d00 * d11 - d01 * d01);
	v = (d11 * d20 - d01 * d21) * denom;
	w = (d00 * d21 - d01 * d20) * denom;
	u = 1.0f - v - w;
}

float Atlast::LinearAlgebra::VectorMath::signed_distance(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::Vectors::Vector3<float> vertex)
{
	return (plane_normal.x * vertex.x + plane_normal.y * vertex.y + plane_normal.z * vertex.z - dotf(plane_normal, point_on_plane));
}
__host__ __device__ bool Atlast::LinearAlgebra::VectorMath::line_clip_against_plane(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::Vectors::Vector3<float> p1, Atlast::Vectors::Vector3<float> p2, Atlast::Vectors::Vector3<float>& new_vertice)
{
	float j1 = dotf(p1, plane_normal);
	float j2 = dotf(p2, plane_normal);

	float g = -dotf(plane_normal, point_on_plane);
	float t = (-g - j1) / (j2 - j1);
	if (t <= 0 || t >= 1) return false;
	new_vertice = p1 + Atlast::Vectors::Vector3<float>(t) * (p2 - p1);
	return true;
}


// intersections.cuh 

__host__ __device__ void Atlast::Algorithms::point_triangle(Atlast::Vectors::Vector3<float> p, Atlast::Vectors::Vector3<float> a, Atlast::Vectors::Vector3<float> b, Atlast::Vectors::Vector3<float> c, float& u, float& v, float& w)
{
	Atlast::Vectors::Vector3<float> v0 = b - a, v1 = c - a, v2 = p - a;
	float den = 1.0f / (v0.x * v1.y - v1.x * v0.y);
	v = (v2.x * v1.y - v1.x * v2.y) * den;
	w = (v0.x * v2.y - v2.x * v0.y) * den;
	u = 1.0f - v - w;

}
__host__ __device__ bool Atlast::Algorithms::ray_triangle(Atlast::Vectors::Vector3<float> ray_origin, Atlast::Vectors::Vector3<float> ray_direction, Atlast::Vectors::Vector3<float> v0, Atlast::Vectors::Vector3<float> v1, Atlast::Vectors::Vector3<float> v2, float& t, float& u, float& v)
{
	Atlast::Vectors::Vector3<float> v0v1 = v1 - v0;
	Atlast::Vectors::Vector3<float> v0v2 = v2 - v0;
	Atlast::Vectors::Vector3<float> pvec = Atlast::LinearAlgebra::VectorMath::crossf(ray_direction, v0v2);
	float det = Atlast::LinearAlgebra::VectorMath::dotf(v0v1, pvec);
	if (fabs(det) < 0.00001f) return false;
	float invDet = 1.0f / det;

	Atlast::Vectors::Vector3<float> tvec = ray_origin - v0;
	u = Atlast::LinearAlgebra::VectorMath::dotf(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	Atlast::Vectors::Vector3<float> qvec = Atlast::LinearAlgebra::VectorMath::crossf(tvec, v0v1);
	v = Atlast::LinearAlgebra::VectorMath::dotf(ray_direction, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;


	t = Atlast::LinearAlgebra::VectorMath::dotf(v0v2, qvec) * invDet;
	if (t < 0) return false;

	return true;
}
