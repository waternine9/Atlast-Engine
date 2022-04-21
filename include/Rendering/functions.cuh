#pragma once
#include "classes.cuh"

namespace Atlast
{
	namespace Rendering
	{

		namespace Device
		{
			__global__ void rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, float* unique_verts, float* uv, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count);
			
			__device__ bool triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, Atlast::RenderingClasses::PopulatedTriangle3D& triangle, Atlast::RenderingClasses::PopulatedTriangle3D& triangle1, Atlast::RenderingClasses::PopulatedTriangle3D& triangle2);
		}
		namespace Cpu
		{
			void texture_fill(Atlast::RenderingClasses::Texture2D& tex, const char* texture_directory, bool normal);
			void rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, float *depth_buffer, float* unique_verts, float* unique_uv, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size);
			void imshow(const char* window_name, unsigned char* buffer, Vector2<unsigned int> resolution, Vector2<unsigned int> stretch_to, float gaussian);
		}
	}
}