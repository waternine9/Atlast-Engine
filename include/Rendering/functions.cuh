#include "classes.cuh"
using namespace Atlast::LinearAlgebra::VectorMath;
using namespace Atlast::Algorithms;
using namespace Atlast::RenderingClasses;

namespace Atlast
{
	namespace Rendering
	{
		
		namespace Device
		{
			__global__ void render_buffer(unsigned char* buffer, Triangle3D* triangles, Camera camera);
		}
		namespace Cpu
		{
			void texture_fill(Texture2D& tex, const char* texture_directory, bool normal);
		}
	}
}