#pragma once
#include "classes.cuh"

namespace Atlast
{
	namespace Rendering
	{

		namespace Device
		{
			__global__ void rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, Atlast::RenderingClasses::PopulatedTriangle3D triangle, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count);
			__global__ void lazy_rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, float* unique_verts, float* uv, Atlast::RenderingClasses::Triangle3D* triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count);
			__host__ __device__ bool triangle_clip_against_plane(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::RenderingClasses::PopulatedTriangle3D& triangle, Atlast::RenderingClasses::PopulatedTriangle3D& triangle1, Atlast::RenderingClasses::PopulatedTriangle3D& triangle2);
			__device__ Atlast::Vectors::Vector3<float> tbn_normals(Atlast::Vectors::Vector3<float> orig_normal, Atlast::Vectors::Vector3<float> normal, Atlast::RenderingClasses::PopulatedTriangle3D out_tri);
		}
		namespace Cpu
		{
			void texture_fill(Atlast::RenderingClasses::Texture2D& tex, const char* texture_directory, bool normal);
			void rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* dev_buffer, float *depth_buffer, Atlast::RenderingClasses::Scene &scene);
		}
	}
}