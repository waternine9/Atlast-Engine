#include "atlast.cuh"
using namespace Atlast;
using namespace Atlast::Vectors;

int main()
{
	Vector3<float> vec1(2.0f, 2.0f, 2.0f);
	vec1 += Vector3<float>(2.0f, 2.0f, 2.0f) * 2.0f;
	printf("%f %f %f\n", vec1.x, vec1.y, vec1.z);
	return 0;
}