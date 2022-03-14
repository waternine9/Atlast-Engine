#include "fastalg.cuh"
__host__ __device__ float Atlast::Algorithms::fast_isqrt(float _X)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = _X * 0.5F;
	y = _X;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));

	return y;
}