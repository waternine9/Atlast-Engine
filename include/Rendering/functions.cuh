#pragma once
#include "classes.cuh"

namespace Atlast
{
	namespace Rendering
	{

		namespace Device
		{
			__global__ void render_buffer(unsigned char* buffer, Atlast::RenderingClasses::Triangle3D* triangles, unsigned int triangles_count, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Light* lights, unsigned int light_count, Atlast::RenderingClasses::Camera camera, float* boundings, unsigned int bounding_count, bool transparent, Vector2<unsigned int> offset, Vector2<unsigned int> resolution);
			__global__ void rasterize_line(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Light* lights, unsigned int light_count);
			__device__ bool device_bvh(Atlast::RenderingClasses::Triangle3D* triangles, Vector3<float> ray_direction, Vector3<float> ray_origin, Atlast::RenderingClasses::Triangle3D& out_tri, float* boundings, int bounding_count, float& out_t, float& out_u, float& out_v);
			__global__ void clip_triangles(Atlast::RenderingClasses::Triangle3D* triangles, unsigned int triangles_size);
			__device__ bool triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, Atlast::RenderingClasses::Triangle3D& triangle, Atlast::RenderingClasses::Triangle3D& triangle1, Atlast::RenderingClasses::Triangle3D& triangle2);
		}
		namespace Cpu
		{
			void texture_fill(Atlast::RenderingClasses::Texture2D& tex, const char* texture_directory, bool normal);
			bool triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, Atlast::RenderingClasses::Triangle3D& triangle, Atlast::RenderingClasses::Triangle3D& triangle1, Atlast::RenderingClasses::Triangle3D& triangle2);
			bool clip_triangles(std::vector<Atlast::RenderingClasses::Triangle3D> &triangles);
			void render_buffer(Vector2<unsigned int> resolution, Vector2<unsigned int> full_resolution, Atlast::RenderingClasses::Camera camera, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, Atlast::RenderingClasses::Triangle3D* dev_triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size, float* dev_boundings, unsigned int boundings_size);
			void rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, float *depth_buffer, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size);
			void imshow(const char* window_name, unsigned char* buffer, Vector2<unsigned int> resolution, Vector2<unsigned int> stretch_to, float gaussian);
		}
	}
}