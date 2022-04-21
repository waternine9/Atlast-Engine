#include "functions.cuh"
#include <opencv2/opencv.hpp>
// functions.cuh

using namespace Atlast::RenderingClasses;

namespace Atlast
{
	namespace Rendering
	{
		void Cpu::texture_fill(Texture2D& tex, const char* texture_directory, bool normal)
		{
			cv::Mat cv_tex = cv::imread(texture_directory);
			cv::resize(cv_tex, cv_tex, cv::Size(1024, 1024));
			cv_tex.convertTo(cv_tex, CV_8UC3);
			cv::cvtColor(cv_tex, cv_tex, cv::COLOR_BGR2RGB);
			for (int y = 0;y < 1024;y++)
			{
				for (int x = 0;x < 1024;x++)
				{
					cv::Vec3b& at = cv_tex.at<cv::Vec3b>(cv::Point(y, x));
					if (!normal)
					{
						tex.data[y][x][0] = at.val[0];
						tex.data[y][x][1] = at.val[1];
						tex.data[y][x][2] = at.val[2];
					}
					else
					{
						tex.normal_data[y][x][0] = (((float)at.val[0]) / 255 - 0.5f) * 2;
						tex.normal_data[y][x][1] = (((float)at.val[1]) / 255 - 0.5f) * 2;
						tex.normal_data[y][x][2] = (((float)at.val[2]) / 255 - 0.5f) * 2;
					}
				}
			}
		}
		
		
		__global__ void Device::rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, float* unique_verts, float* uv, Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count)
		{
			int i = threadIdx.x;
			int j = blockIdx.x;
			unsigned int triangle_idx = i + j * 256;
			if (triangle_idx < triangles_size)
			{
				Triangle3D& triangle = triangles[triangle_idx];

				unsigned int vert_idx = triangle.i0 * 3;
				Vector3<float> tri0(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);
				vert_idx = triangle.i1 * 3;
				Vector3<float> tri1(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);
				vert_idx = triangle.i2 * 3;
				Vector3<float> tri2(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);

				vert_idx = triangle.uv0 * 2;
				Vector2<float> uv0(uv[vert_idx], uv[vert_idx + 1]);
				vert_idx = triangle.uv1 * 2;
				Vector2<float> uv1(uv[vert_idx], uv[vert_idx + 1]);
				vert_idx = triangle.uv2 * 2;
				Vector2<float> uv2(uv[vert_idx], uv[vert_idx + 1]);
				tri0 = LinearAlgebra::VectorMath::rotatef(tri0 + translate, rotate, Vector3<float>(0));
				tri1 = LinearAlgebra::VectorMath::rotatef(tri1 + translate, rotate, Vector3<float>(0));
				tri2 = LinearAlgebra::VectorMath::rotatef(tri2 + translate, rotate, Vector3<float>(0));
				
				Vector3<float> triangle_normal = normalizef(crossf(tri2 - tri0, tri1 - tri0));

				PopulatedTriangle3D populated_triangle;
				populated_triangle.vertices[0] = tri0;
				populated_triangle.vertices[2] = tri1;
				populated_triangle.vertices[1] = tri2;
				PopulatedTriangle3D new_tris[2] = { populated_triangle };
				Device::triangle_clip_against_plane(Vector3<float>(0, 0, -0.1), Vector3<float>(0, 0, -1), populated_triangle, new_tris[0], new_tris[1]); // Stores clipped triangles in the last two parameters

				for (int sub_triangle_idx = 0;sub_triangle_idx < 2;sub_triangle_idx++)
				{
					PopulatedTriangle3D bruh = new_tris[sub_triangle_idx];
					Vector3<float> v0, v1, v2;
					if (bruh.vertices[0].x == 0) continue;
					
					v0 = bruh.vertices[0];
					v1 = bruh.vertices[1];
					v2 = bruh.vertices[2];

					float y11 = (v0.y / v0.z) * (full_resolution.y * 0.5) + (full_resolution.y * 0.5);
					float y21 = (v1.y / v1.z) * (full_resolution.y * 0.5) + (full_resolution.y * 0.5);
					float y31 = (v2.y / v2.z) * (full_resolution.y * 0.5) + (full_resolution.y * 0.5);

					float x1 = (v0.x / v0.z) * (full_resolution.x * 0.5) + (full_resolution.x * 0.5);
					float x2 = (v1.x / v1.z) * (full_resolution.x * 0.5) + (full_resolution.x * 0.5);
					float x3 = (v2.x / v2.z) * (full_resolution.x * 0.5) + (full_resolution.x * 0.5);
					const Vector4<int> bounding(
						(int)min(min(x1, x2), x3),
						(int)max(max(x1, x2), x3),
						(int)min(min(y11, y21), y31),
						(int)max(max(y11, y21), y31)
					);
					if ((bounding.x < 0 && bounding.y < 0) || (bounding.x >= full_resolution.x && bounding.y >= full_resolution.x) || (bounding.z < 0 && bounding.w < 0) || (bounding.z >= full_resolution.x && bounding.w >= full_resolution.x)) break; 
					if (!(bounding.x == bounding.y && bounding.z == bounding.w))
					{
						Vector3<float> B0 = Vector3<float>(x2, y21, 0) - Vector3<float>(x1, y11, 0), B1 = Vector3<float>(x3, y31, 0) - Vector3<float>(x1, y11, 0);
						float d00 = dotf(B0, B0);
						float d01 = dotf(B0, B1);
						float d11 = dotf(B1, B1);
						float denom = 1.0f / (d00 * d11 - d01 * d01);
						for (int iy = bounding.z;iy < bounding.w + 1;iy++)
						{
							if (iy >= full_resolution.y || iy < 0) continue;
							for (int ix = bounding.x;ix < bounding.y + 1;ix++)
							{
								if (ix >= full_resolution.x || ix < 0) continue;
								float u, v, w;
								Vector3<float> B2 = Vector3<float>(ix, iy, 0) - Vector3<float>(x1, y11, 0);
								float d20 = dotf(B2, B0);
								float d21 = dotf(B2, B1);
								v = (d11 * d20 - d01 * d21) * denom;
								w = (d00 * d21 - d01 * d20) * denom;
								if (v >= 0 && w >= 0 && v + w < 1.0f)
								{
									u = 1.0f - v - w;
									Vector3<float> frag_coord = (Vector3<float>(x1, y11, v0.z * (full_resolution.y * 0.5) + (full_resolution.y * 0.5))) * u + (Vector3<float>(x2, y21, v1.z * (full_resolution.y * 0.5) + (full_resolution.y * 0.5))) * v + (Vector3<float>(x2, y31, v2.z * (full_resolution.y * 0.5) + (full_resolution.y * 0.5))) * w;

									if (depth_buffer[ix + iy * full_resolution.x] == 0.0 || frag_coord.z > depth_buffer[ix + iy * full_resolution.x])
									{
										if (frag_coord.z > -0.1) continue;
										Vector2<float> uv = uv0 * u + uv1 * v + uv2 * w;
										Material material = materials[triangle.material_idx];
										Vector3<unsigned char> triangle_color = material.color;
										if (material.textured)
										{
											if (textures[material.texture_idx].albedo) triangle_color = Vector3<unsigned char>(
												textures[material.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][2],
												textures[material.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][1],
												textures[material.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][0]
												);
										}
										

										Vector3<float> wsfc = (v0 * u + v1 * v + v2 * w);
										Vector3<float> light_dir = normalizef(wsfc * -1.0f);
										float diffuse = max(0.0f, dotf(triangle_normal, light_dir));
										float specular = max(0.0f, powf(dotf(reflectf(light_dir, triangle_normal), light_dir * -1.0f ) * material.reflective, 32));
										float lums = diffuse + specular;
										buffer[(int)(ix + iy * full_resolution.x) * 3] = (int)triangle_color.x * lums < 255 ? (int)triangle_color.x * lums : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = (int)triangle_color.y * lums < 255 ? (int)triangle_color.y * lums : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = (int)triangle_color.z * lums < 255 ? (int)triangle_color.z * lums : 255;
										depth_buffer[ix + iy * full_resolution.x] = frag_coord.z;
									}
								}
							}
						}
					}
				}
				
			}
		}
		
		void Cpu::rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, float* depth_buffer, float* unique_verts, float* unique_uv, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Material* materials, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size)
		{
			Device::rasterize_buffer<<<max(triangles_size >> 8, 1), 256>>>(translate, rotate, full_resolution, dev_buffer, depth_buffer, unique_verts, unique_uv, triangles, triangles_size, dev_textures, materials, dev_lights, lights_size);
			cudaDeviceSynchronize();
		}
		bool Device::triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, PopulatedTriangle3D& triangle, PopulatedTriangle3D& triangle1, PopulatedTriangle3D& triangle2)
		{
			Vector3<float> new_point;
			Vector3<float> inside_points[4];
			Vector3<float> outside_points[4];
			int outside_point_count = 0;
			int inside_point_count = 0;
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[0]) < 0)
			{
				outside_point_count++;
				outside_points[outside_point_count - 1] = triangle.vertices[0];
			}
			else
			{
				inside_point_count++;
				inside_points[inside_point_count - 1] = triangle.vertices[0];
			}
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[1]) < 0)
			{
				outside_point_count++;
				outside_points[outside_point_count - 1] = triangle.vertices[1];
			}
			else
			{
				inside_point_count++;
				inside_points[inside_point_count - 1] = triangle.vertices[1];
			}
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[2]) < 0)
			{
				outside_point_count++;
				outside_points[outside_point_count - 1] = triangle.vertices[2];
			}
			else
			{
				inside_point_count++;
				inside_points[inside_point_count - 1] = triangle.vertices[2];
			}
			if (inside_point_count == 0)
			{
				return true;
			}

			if (inside_point_count == 3)
			{
				return false;
			}

			if (inside_point_count == 1 && outside_point_count == 2)
			{
				triangle1.vertices[0] = inside_points[0];
				Vector3<float> new_point;
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[0], new_point);
				triangle1.vertices[1] = new_point;
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[1], new_point);
				triangle1.vertices[2] = new_point;

				return true;
			}

			if (inside_point_count == 2 && outside_point_count == 1)
			{
				triangle1.vertices[0] = inside_points[0];
				triangle1.vertices[1] = inside_points[1];
				Vector3<float> new_point;
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[0], new_point);
				triangle1.vertices[2] = new_point;

				triangle2.vertices[0] = inside_points[1];
				triangle2.vertices[1] = triangle1.vertices[2];
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[1], outside_points[0], new_point);
				triangle2.vertices[2] = new_point;

				return true;
			}
			return false;
		}
		
		
		void Cpu::imshow(const char* window_name, unsigned char* buffer, Vector2<unsigned int> resolution, Vector2<unsigned int> stretch_to, float gaussian)
		{

			cv::Mat canvas = cv::Mat::zeros(cv::Size(resolution.x, resolution.y), CV_8UC3);
			for (int y = 0;y < canvas.rows;y++)
			{
				for (int x = 0;x < canvas.cols;x++)
				{

					cv::Vec3b& at = canvas.at<cv::Vec3b>(y, x);
					at.val[0] = buffer[(x + y * canvas.cols) * 3];
					at.val[1] = buffer[(x + y * canvas.cols) * 3 + 1];
					at.val[2] = buffer[(x + y * canvas.cols) * 3 + 2];
					buffer[(x + y * canvas.cols) * 3] = 0;
					buffer[(x + y * canvas.cols) * 3 + 1] = 0;
					buffer[(x + y * canvas.cols) * 3 + 2] = 0;
				}
			}
			cv::resize(canvas, canvas, cv::Size(stretch_to.x, stretch_to.y));
			cv::GaussianBlur(canvas, canvas, cv::Size(1, 1), gaussian);
			cv::imshow(window_name, canvas);
		}
		
	
	}
	// classes.cuh 
	namespace RenderingClasses
	{
		int HelperFuncs::len(std::string str)
		{
			int length = 0;
			for (int i = 0; str[i] != '\0'; i++)
			{
				length++;
			}
			return length;

		}
		void HelperFuncs::split(std::string str, char seperator, std::vector<std::string>& substrings)
		{
			int currIndex = 0, i = 0;
			int startIndex = 0, endIndex = 0;
			while (i <= len(str))
			{
				if (str[i] == seperator || i == len(str))
				{
					endIndex = i;
					std::string subStr = "";
					subStr.append(str, startIndex, endIndex - startIndex);
					substrings.push_back(subStr);
					currIndex += 1;
					startIndex = endIndex + 1;
				}
				i++;
			}
		}
	}
}