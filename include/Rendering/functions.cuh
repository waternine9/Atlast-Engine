#pragma once
#include "classes.cuh"

namespace Atlast
{
	namespace Rendering
	{

		namespace Device
		{
			__global__ void render_buffer(unsigned char* buffer, Atlast::RenderingClasses::Triangle3D* triangles, unsigned int triangles_count, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Light* lights, unsigned int light_count, Atlast::RenderingClasses::Camera camera, bool transparent);

		}
		namespace Cpu
		{
			void texture_fill(Atlast::RenderingClasses::Texture2D& tex, const char* texture_directory, bool normal);
			void render_buffer(unsigned char* buffer, std::vector<Atlast::RenderingClasses::Triangle3D> triangles, std::vector<Atlast::RenderingClasses::Texture2D> textures, std::vector<Atlast::RenderingClasses::Light> lights, Atlast::RenderingClasses::Camera camera, Vector2<unsigned int> resolution, bool transparent);
			void imshow(const char* window_name, unsigned char* buffer, Vector2<unsigned int> resolution);
		}
	}
}