#include "linalg.cuh"

__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Vector3<float> _X, const Vector3<float> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y + _X.z * _Y.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Vector2<float> _X, const Vector2<float> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dotf(const Vector4<float> _X, const Vector4<float> _Y)
{
	return (_X.x * _Y.y + _X.y * _Y.y + _X.z * _Y.z + _X.w * _Y.w);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Vector3<double> _X, const Vector3<double> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y + _X.z * _Y.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Vector2<double> _X, const Vector2<double> _Y)
{
	return (_X.x * _Y.x + _X.y * _Y.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::dot(const Vector4<double> _X, const Vector4<double> _Y)
{
	return (_X.x * _Y.y + _X.y * _Y.y + _X.z * _Y.z + _X.w * _Y.w);
}
__host__ __device__ Vector3<float> Atlast::LinearAlgebra::VectorMath::crossf(const Vector3<float> _X, const Vector3<float> _Y)
{
	return Vector3<float>(
		_X.y * _Y.z - _X.z * _Y.y,
		-(_X.x * _Y.z - _X.z * _Y.x),
		_X.x * _Y.y - _X.y * _Y.x
		);
}
__host__ __device__ Vector3<double> Atlast::LinearAlgebra::VectorMath::cross(const Vector3<double> _X, const Vector3<double> _Y)
{
	return Vector3<double>(
		_X.y * _Y.z - _X.z * _Y.y,
		-(_X.x * _Y.z - _X.z * _Y.x),
		_X.x * _Y.y - _X.y * _Y.x
		);
}
__host__ __device__ Vector3<float> Atlast::LinearAlgebra::VectorMath::reflectf(Vector3<float> _X, Vector3<float> _NORMAL)
{
	return _X - _NORMAL * dotf(_NORMAL, _X) * 2.0f;
}
__host__ __device__ Vector3<double> Atlast::LinearAlgebra::VectorMath::reflect(Vector3<double> _X, Vector3<double> _NORMAL)
{
	return _X - _NORMAL * dot(_NORMAL, _X) * 2.0f;
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Vector2<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Vector3<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ float Atlast::LinearAlgebra::VectorMath::lengthf(Vector4<float> _X)
{
	return sqrtf(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Vector2<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Vector3<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ double Atlast::LinearAlgebra::VectorMath::lengthf(Vector4<double> _X)
{
	return sqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Vector2<float> Atlast::LinearAlgebra::VectorMath::normalizef(Vector2<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ Vector3<float> Atlast::LinearAlgebra::VectorMath::normalizef(Vector3<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ Vector4<float> Atlast::LinearAlgebra::VectorMath::normalizef(Vector4<float> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Vector2<double> Atlast::LinearAlgebra::VectorMath::normalize(Vector2<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y);
}
__host__ __device__ Vector3<double> Atlast::LinearAlgebra::VectorMath::normalize(Vector3<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z);
}
__host__ __device__ Vector4<double> Atlast::LinearAlgebra::VectorMath::normalize(Vector4<double> _X)
{
	return _X * Atlast::Algorithms::fast_isqrt(_X.x * _X.x + _X.y * _X.y + _X.z * _X.z + _X.w * _X.w);
}
__host__ __device__ Vector3<float> Atlast::LinearAlgebra::VectorMath::rotatef(Vector3<float> v, Vector3<float> rotate_by, Vector3<float> center)
{
	v = v - center;
	float SINF = sinf(rotate_by.z);
	float COSF = cosf(rotate_by.z);
	float output2[1][3] = { { 0, 0, 0 } };
	float input1[1][3];
	float input2[3][3];
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
	v = Vector3<float>((float)output2[0][0], (float)output2[0][1], (float)output2[0][2]);
	SINF = sinf(rotate_by.y);
	COSF = cosf(rotate_by.y);
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
	v = Vector3<float>((float)output[0][0], (float)output[0][1], (float)output[0][2]);
	SINF = sinf(rotate_by.x);
	COSF = cosf(rotate_by.x);
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
	v = Vector3<float>((float)output4[0][0], (float)output4[0][1], (float)output4[0][2]);

	v = v + center;
	return v;
}

// intersections.cuh 

__host__ __device__ void Atlast::Algorithms::point_triangle(Vector3<float> p, Vector3<float> a, Vector3<float> b, Vector3<float> c, float& u, float& v, float& w)
{
	Vector3<float> v0 = b - a, v1 = c - a, v2 = p - a;
	float den = 1.0f / (v0.x * v1.y - v1.x * v0.y);
	v = (v2.x * v1.y - v1.x * v2.y) * den;
	w = (v0.x * v2.y - v2.x * v0.y) * den;
	u = 1.0f - v - w;

}
__host__ __device__ bool Atlast::Algorithms::ray_triangle(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector3<float> v0, Vector3<float> v1, Vector3<float> v2, float& t, float& u, float& v)
{
	Vector3<float> v0v1 = v1 - v0;
	Vector3<float> v0v2 = v2 - v0;
	Vector3<float> pvec = crossf(ray_direction, v0v2);
	float det = dotf(v0v1, pvec);
	if (fabs(det) < 0.00001f) return false;
	float invDet = 1.0f / det;

	Vector3<float> tvec = ray_origin - v0;
	u = dotf(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	Vector3<float> qvec = crossf(tvec, v0v1);
	v = dotf(ray_direction, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;


	t = dotf(v0v2, qvec) * invDet;
	if (t < 0) return false;

	return true;
}
__host__ __device__ bool Atlast::Algorithms::ray_aabb(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector2<Vector3<float>> minmax, float& t)
{
	Vector3<float> inv_direction = Vector3<float>(1.0f) / ray_direction;
	Vector3<float> tMin = (minmax.x - ray_origin) * inv_direction;
	Vector3<float> tMax = (minmax.y - ray_origin) * inv_direction;
	Vector3<float> t1 = Vector3<float>(min(tMin.x, tMax.x), min(tMin.y, tMax.y), min(tMin.z, tMax.z));
	Vector3<float> t2 = Vector3<float>(max(tMin.x, tMax.x), max(tMin.y, tMax.y), max(tMin.z, tMax.z));
	float tNear = max(max(t1.x, t1.y), t1.z);
	float tFar = min(min(t2.x, t2.y), t2.z);
	if (ray_origin.x > minmax.x.x && ray_origin.x < minmax.y.x && ray_origin.y > minmax.x.y && ray_origin.y < minmax.y.y && ray_origin.z > minmax.x.z && ray_origin.z < minmax.y.z)
	{
		t = tFar;
		return true;
	}
	if (tNear > tFar) return false;
	if (tFar < 0) return false;
	t = tNear;
	return true;
}
__host__ __device__ bool Atlast::Algorithms::ray_sphere(Vector3<float> ray_origin, Vector3<float> ray_direction, Vector3<float> center, float radius, float& t)
{
	Vector3<float> oc = ray_origin - center;
	float a = dotf(ray_direction, ray_direction);
	float B = 2.0 * dotf(oc, ray_direction);
	float c = dotf(oc, oc) - radius * radius;
	float discriminant = B * B - 4 * a * c;
	if (discriminant >= 0.0) {
		float numerator = -B - sqrtf(discriminant);
		if (numerator > 0.0) {
			float dist = numerator / (2.0 * a);
			t = dist;
			return true;
		}
	}
	return false;

}