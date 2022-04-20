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
		__device__ bool Device::device_bvh(Atlast::RenderingClasses::Triangle3D* triangles, Vector3<float> ray_direction, Vector3<float> ray_origin, Triangle3D& out_tri, float* boundings, int bounding_count, float& out_t, float& out_u, float& out_v)
		{
			int excluded[400];
			int excluded_count = 0;

			while (excluded_count != bounding_count)
			{
				float t2 = 100000000.0f;
				int start_tri = -1;
				int end_tri = -1;
				int b_idx2 = 0;
				Vector3<float> min_point2;
				Vector3<float> max_point2;
				for (int b_idx = 0;b_idx < bounding_count;b_idx++)
				{
					Vector3<float> min_point(boundings[b_idx * 8], boundings[b_idx * 8 + 1], boundings[b_idx * 8 + 2]);
					Vector3<float> max_point(boundings[b_idx * 8 + 3], boundings[b_idx * 8 + 4], boundings[b_idx * 8 + 5]);
					float temp;
					if (Atlast::Algorithms::ray_aabb(ray_origin, ray_direction, Vector2<Vector3<float>>(min_point, max_point), temp))
					{

						if (temp < t2)
						{
							bool should_continue = false;
							for (int j = 0;j < excluded_count;j++)
							{
								if (b_idx == excluded[j]) should_continue = true;
							}
							if (should_continue) continue;
							start_tri = (int)boundings[b_idx * 8 + 6];
							end_tri = (int)boundings[b_idx * 8 + 7];
							t2 = temp;
							b_idx2 = b_idx;
							min_point2 = min_point;
							max_point2 = max_point;
						}
					}
				}


				if (start_tri == -1)
				{
					return false;
				}
				float t = 100000000.0f, u, v;
				Triangle3D triangle;
				for (int b_idx = 0;b_idx < bounding_count;b_idx++)
				{
					Vector3<float> min_point(boundings[b_idx * 8], boundings[b_idx * 8 + 1], boundings[b_idx * 8 + 2]);
					Vector3<float> max_point(boundings[b_idx * 8 + 3], boundings[b_idx * 8 + 4], boundings[b_idx * 8 + 5]);
					if ((min_point.x > min_point2.x && min_point.y > min_point2.y && min_point.z > min_point2.z) || (max_point.x < max_point2.x && max_point.y < max_point2.y && max_point.z < max_point2.z))
					{
						int start_tri2 = (int)boundings[b_idx * 8 + 6];
						int end_tri2 = (int)boundings[b_idx * 8 + 7];
						for (int t_idx = start_tri2;t_idx <= end_tri2;t_idx++)
						{
							Triangle3D triangle2 = triangles[t_idx];
							float dist, u2, v2;
							if (Atlast::Algorithms::ray_triangle(ray_origin, ray_direction, triangle2.vertices[0], triangle2.vertices[1], triangle2.vertices[2], dist, u2, v2))
							{
								if (dist < t)
								{
									t = dist;
									u = u2;
									v = v2;
									triangle = triangle2;
								}
							}
						}
					}
				}

				for (int t_idx = start_tri;t_idx <= end_tri;t_idx++)
				{
					Triangle3D triangle2 = triangles[t_idx];
					float dist, u2, v2;
					if (Atlast::Algorithms::ray_triangle(ray_origin, ray_direction, triangle2.vertices[0], triangle2.vertices[1], triangle2.vertices[2], dist, u2, v2))
					{
						if (dist < t)
						{
							t = dist;
							u = u2;
							v = v2;
							triangle = triangle2;
						}
					}
				}
				if (t == 100000000.0f)
				{
					excluded[excluded_count] = b_idx2;
					excluded_count++;
					if (excluded_count >= 397) return false;
				}
				else
				{
					out_tri = triangle;
					out_t = t;
					out_u = u;
					out_v = v;

					return true;
				}
			}
		}
		__global__ void Device::render_buffer(unsigned char* buffer, Triangle3D* triangles, unsigned int triangles_count, Texture2D* textures, Light* lights, unsigned int light_count, Camera camera, float* boundings, unsigned int bounding_count, bool transparent, Vector2<unsigned int> offset, Vector2<unsigned int> resolution)
		{
			float i = threadIdx.x + offset.x;
			float j = blockIdx.x + offset.y;
			float resolution_x = resolution.x;
			int r_idx = (i + j * resolution_x) * 3;
			int g_idx = (i + j * resolution_x) * 3 + 1;
			int b_idx = (i + j * resolution_x) * 3 + 2;

			if (!transparent)
			{
				buffer[r_idx] = camera.bg_col.x;
				buffer[g_idx] = camera.bg_col.y;
				buffer[b_idx] = camera.bg_col.z;
			}

			Vector3<float> ray_pos = camera.pos;
			float inv_resolution_x = 1.0f / resolution_x;
			Vector3<float> ray_vec = rotatef(
				normalizef(Vector3<float>(
					(i * inv_resolution_x) - 0.5f,
					-((j * inv_resolution_x) - 0.5f),
					1.0f
					)),
				Vector3<float>(camera.rot.x, 0, 0),
				Vector3<float>(0.0f)
			);
			ray_vec = rotatef(ray_vec, Vector3<float>(0, camera.rot.y, 0), Vector3<float>(0.0f));
			Triangle3D cur_tri;
			float closest_t = 1e+10f;
			float closest_u;
			float closest_v;

			float closest_w;
			if (device_bvh(triangles, ray_vec, ray_pos, cur_tri, boundings, bounding_count, closest_t, closest_u, closest_v))
			{
				Vector3<float> intersection = ray_pos + ray_vec * closest_t;
				barycentric(ray_pos + ray_vec * closest_t, cur_tri.vertices[0], cur_tri.vertices[1], cur_tri.vertices[2], closest_u, closest_v, closest_w);
				if (cur_tri.textured)
				{
					if (cur_tri.smooth_shading)
					{
						cur_tri.normal = (cur_tri.normals.x * closest_u + cur_tri.normals.y * closest_v + cur_tri.normals.z * (1.0f - closest_u - closest_v));
					}
					if (dotf(ray_vec * closest_t, cur_tri.normal) < 0)
					{
						cur_tri.normal = cur_tri.normal * -1.0f;
					}
					Vector2<float> tex_coord = (cur_tri.uv.x * closest_u + cur_tri.uv.y * closest_v + cur_tri.uv.z * (1.0f - closest_u - closest_v)) * 1024;
					tex_coord.x = ((int)(tex_coord.x * textures[cur_tri.texture_idx].repeating + textures[cur_tri.texture_idx].offset.x) % 1024);
					tex_coord.y = ((int)(tex_coord.y * textures[cur_tri.texture_idx].repeating + textures[cur_tri.texture_idx].offset.y) % 1024);

					if (textures[cur_tri.texture_idx].albedo)
					{
						cur_tri.color = Vector3<unsigned char>(textures[cur_tri.texture_idx].data[(int)tex_coord.y][(int)tex_coord.x][2], textures[cur_tri.texture_idx].data[(int)tex_coord.y][(int)tex_coord.x][1], textures[cur_tri.texture_idx].data[(int)tex_coord.y][(int)tex_coord.x][0]);
					}

					if (textures[cur_tri.texture_idx].normal)
					{
						Vector3<float> normal(textures[cur_tri.texture_idx].normal_data[(int)tex_coord.y][(int)tex_coord.x][0], textures[cur_tri.texture_idx].normal_data[(int)tex_coord.y][(int)tex_coord.x][1], textures[cur_tri.texture_idx].normal_data[(int)tex_coord.y][(int)tex_coord.x][2]);
						Vector3<float> tt;
						Vector3<float> b;

						Vector3<float> edge1 = cur_tri.vertices[1] - cur_tri.vertices[0];
						Vector3<float> edge2 = cur_tri.vertices[2] - cur_tri.vertices[0];

						Vector2<float> deltaUV1 = cur_tri.uv.y - cur_tri.uv.x;
						Vector2<float> deltaUV2 = cur_tri.uv.z - cur_tri.uv.x;

						float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

						tt.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
						tt.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
						tt.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

						b.x = -f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
						b.y = -f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
						b.z = -f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
						tt = normalizef(tt - Vector3<float>(dotf(tt, cur_tri.normal)) * cur_tri.normal);
						b = normalizef(crossf(tt, cur_tri.normal));
						float output7[1][3] = { { 0, 0, 0 } };
						float input2[3][3];
						float input1[1][3];
						input2[0][0] = tt.x;
						input2[0][1] = tt.y;
						input2[0][2] = tt.z;
						input2[1][0] = b.x;
						input2[1][1] = b.y;
						input2[1][2] = b.z;
						input2[2][0] = cur_tri.normal.x;
						input2[2][1] = cur_tri.normal.y;
						input2[2][2] = cur_tri.normal.z;
						input1[0][0] = normal.x;
						input1[0][1] = normal.y;
						input1[0][2] = normal.z;
						for (int _ = 0;_ < 1;_++)
							for (int Y = 0;Y < 3;Y++)
								for (int k = 0;k < 3;k++)
								{
									output7[_][Y] += input1[_][k] * input2[k][Y];
								}
						cur_tri.normal = normalizef(Vector3<float>((float)output7[0][0], (float)output7[0][1], (float)output7[0][2]));
					}
				}

				float avg_lums = 0.0f;
				float avg_spec = 0.0f;
				Vector3<float> avg_col;
				for (int light_idx = 0;light_idx < light_count;light_idx++)
				{
					Vector3<float> itol = (lights[light_idx].pos - intersection);
					float itol_mag = lengthf(itol);
					float intensity = max(-dotf(normalizef(itol), cur_tri.normal), 0.0f);
					intensity *= lights[light_idx].intensity / itol_mag;
					avg_col = avg_col + Vector3<float>(lights[light_idx].color.x * intensity, lights[light_idx].color.y * intensity, lights[light_idx].color.z * intensity);
					Vector3<float> ray_to_sun_vec = normalizef(lights[light_idx].pos - intersection);
					Vector3<float> ray_to_sun_origin = intersection + ray_to_sun_vec * Vector3<float>(0.001f);
					int shadowHit = 1;
					float t2 = itol_mag;
					float temp, temp_u, temp_v;
					Triangle3D tri_temp;
					if (device_bvh(triangles, ray_to_sun_vec, ray_to_sun_origin, tri_temp, boundings, bounding_count, temp, temp_u, temp_v))// (Intersections::device_raycast(triangles, ray_to_sun, tri_temp, temp, boundings, bounding_count))
					{
						if (temp < t2) shadowHit = 0;
					}
					avg_lums += intensity * shadowHit;
					avg_spec += pow(max(dotf(ray_vec, reflectf(ray_vec, cur_tri.normal)), 0.0f), 256);
				}
				avg_col = normalizef(avg_col);
				buffer[r_idx] = min(cur_tri.color.x * (avg_lums + avg_spec) * avg_col.x, 255.0f);
				buffer[g_idx] = min(cur_tri.color.y * (avg_lums + avg_spec) * avg_col.y, 255.0f);
				buffer[b_idx] = min(cur_tri.color.z * (avg_lums + avg_spec) * avg_col.z, 255.0f);
			}
		}
		__global__ void Device::rasterize_line(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Light* lights, unsigned int light_count)
		{
			int i = threadIdx.x;
			int j = blockIdx.x;
			unsigned int triangle_idx = i + j * 256;
			if (triangle_idx < triangles_size)
			{
				Triangle3D& triangle = triangles[triangle_idx];

				triangle.vertices[0] = LinearAlgebra::VectorMath::rotatef(triangle.vertices[0] + translate, rotate, Vector3<float>(0));
				triangle.vertices[1] = LinearAlgebra::VectorMath::rotatef(triangle.vertices[1] + translate, rotate, Vector3<float>(0));
				triangle.vertices[2] = LinearAlgebra::VectorMath::rotatef(triangle.vertices[2] + translate, rotate, Vector3<float>(0));
				
				triangle.normal = LinearAlgebra::VectorMath::rotatef(triangle.normal, rotate, Vector3<float>(0));

				Triangle3D new_tris[2] = { triangle };
				triangle_clip_against_plane(Vector3<float>(0, 0, -0.1), Vector3<float>(0, 0, -1), triangle, new_tris[0], new_tris[1]); // Stores clipped triangles in the last two parameters

				for (int sub_triangle_idx = 0;sub_triangle_idx < 2;sub_triangle_idx++)
				{
					Triangle3D bruh = new_tris[sub_triangle_idx];
					if (bruh.vertices[0].x == 0) continue;
					Vector3<float> v0 = bruh.vertices[0];
					Vector3<float> v1 = bruh.vertices[1];
					Vector3<float> v2 = bruh.vertices[2];

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
										Vector2<float> uv = triangle.uv.x * u + triangle.uv.y * v + triangle.uv.z * w;
										if (triangle.smooth_shading) triangle.normal = (triangle.normals.x * u + triangle.normals.y * v + triangle.normals.z * w) * -1;
										if (triangle.textured)
										{
											if (textures[triangle.texture_idx].albedo) triangle.color = Vector3<unsigned char>(
												textures[triangle.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][2],
												textures[triangle.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][1],
												textures[triangle.texture_idx].data[(int)(uv.y * 1024)][(int)(uv.x * 1024)][0]
												);
										}


										Vector3<float> wsfc = (v0 * u + v1 * v + v2 * w);
										Vector3<float> light_dir = normalizef(wsfc * -1.0f);
										float diffuse = max(0.0f, dotf(triangle.normal, light_dir));
										float lums = diffuse;
										buffer[(int)(ix + iy * full_resolution.x) * 3] = triangle.color.x * lums < 255 ? triangle.color.x * lums : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = triangle.color.y * lums < 255 ? triangle.color.y * lums : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = triangle.color.z * lums < 255 ? triangle.color.z * lums : 255;
										depth_buffer[ix + iy * full_resolution.x] = frag_coord.z;
									}
								}
							}
						}
					}
				}
				/*
				// Deltas
				const int Dx12 = x1 - x2;
				const int Dx23 = x2 - x3;
				const int Dx31 = x3 - x1;

				const int Dy12 = y11 - y21;
				const int Dy23 = y21 - y31;
				const int Dy31 = y31 - y11;

				const int FDX12 = Dx12 << 4;
				const int FDX23 = Dx23 << 4;
				const int FDX31 = Dx31 << 4;

				const int FDY12 = Dy12 << 4;
				const int FDY23 = Dy23 << 4;
				const int FDY31 = Dy31 << 4;

				float C1 = Dy12 * x1 - Dx12 * y11;
				float C2 = Dy23 * x2 - Dx23 * y21;
				float C3 = Dy31 * x3 - Dx31 * y31;
				
				if (Dy12 < 0 || (Dy12 == 0 && Dx12 > 0)) C1++;
				if (Dy23 < 0 || (Dy23 == 0 && Dx23 > 0)) C2++;
				if (Dy31 < 0 || (Dy31 == 0 && Dx31 > 0)) C3++;
				Vector3<float> B0 = Vector3<float>(x2, y21, 0) - Vector3<float>(x1, y11, 0), B1 = Vector3<float>(x3, y31, 0) - Vector3<float>(x1, y11, 0);
				float d00 = dotf(B0, B0);
				float d01 = dotf(B0, B1);
				float d11 = dotf(B1, B1);
				float denom = 1.0 / (d00 * d11 - d01 * d01);
				for (int _y = bounding.z;_y < bounding.w;_y += 8)
				{
					if (_y >= 0 && _y < full_resolution.y - 8)
					{

						for (int _x = bounding.x;_x < bounding.y;_x += 8)
						{
							if (_x >= 0 && _x < full_resolution.x - 8)
							{
								int x0 = _x << 4;
								int x1 = (_x + 7) << 4;
								int y0 = _y << 4;
								int y1 = (_y + 7) << 4;
								bool a00 = C1 + Dx12 * y0 - Dy12 * x0 > 0;
								bool a10 = C1 + Dx12 * y0 - Dy12 * x1 > 0;
								bool a01 = C1 + Dx12 * y1 - Dy12 * x0 > 0;
								bool a11 = C1 + Dx12 * y1 - Dy12 * x1 > 0;
								int a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);
								bool b00 = C2 + Dx23 * y0 - Dy23 * x0 > 0;
								bool b10 = C2 + Dx23 * y0 - Dy23 * x1 > 0;
								bool b01 = C2 + Dx23 * y1 - Dy23 * x0 > 0;
								bool b11 = C2 + Dx23 * y1 - Dy23 * x1 > 0;
								int b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);
								bool c00 = C3 + Dx31 * y0 - Dy31 * x0 > 0;
								bool c10 = C3 + Dx31 * y0 - Dy31 * x1 > 0;
								bool c01 = C3 + Dx31 * y1 - Dy31 * x0 > 0;
								bool c11 = C3 + Dx31 * y1 - Dy31 * x1 > 0;
								int c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);
								if (a == 0xF && b == 0xF && c == 0xF)
								{
									for (int iy = _y;iy < _y + 8;iy++)
									{
										for (int ix = _x;ix < _x + 8;ix++)
										{
											float u, v, w;
											Vector3<float> B2 = Vector3<float>(ix, iy, 0) - Vector3<float>(x1, y11, 0);
											float d20 = dotf(B2, B0);
											float d21 = dotf(B2, B1);
											v = (d11 * d20 - d01 * d21) * denom;
											w = (d00 * d21 - d01 * d20) * denom;
											u = 1.0f - v - w;
											Vector3<float> frag_coord = triangle.vertices[0] * u + triangle.vertices[1] * v + triangle.vertices[2] * w;
											if (depth_buffer[ix + iy * full_resolution.x] == 0 || frag_coord.z > depth_buffer[ix + iy * full_resolution.x])
											{
												Vector2<float> uv = triangle.uv.x * u + triangle.uv.y * v + triangle.uv.z * w;
												triangle.normal = (triangle.normals.x * uv.x + triangle.normals.y * uv.y + triangle.normals.z * (1.0 - uv.x - uv.y)) * -1;
												
												float lums = max(0.0f, dotf(triangle.normal, Vector3<float>(0.333, 0.333, 0.333)));
												buffer[(int)(ix + iy * full_resolution.x) * 3] = triangle.color.x * lums * u;
												buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = triangle.color.y * lums * v;
												buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = triangle.color.z * lums * w;
												depth_buffer[ix + iy * full_resolution.x] = frag_coord.z;
											}
										}
									}
								}
								else
								{
									int CY1 = C1 + Dx12 * y0 - Dy12 * x0;
									int CY2 = C2 + Dx23 * y0 - Dy23 * x0;
									int CY3 = C3 + Dx31 * y0 - Dy31 * x0;
									for (int iy = _y;iy < _y + 8;iy++)
									{
										int CX1 = CY1;
										int CX2 = CY2;
										int CX3 = CY3;

										for (int ix = _x;ix < _x + 8;ix++)
										{
											if (CX1 > 0 && CX2 > 0 && CX3 > 0)
											{
												float u, v, w;
												Vector3<float> B2 = Vector3<float>(ix, iy, 0) - Vector3<float>(x1, y11, 0);
												float d20 = dotf(B2, B0);
												float d21 = dotf(B2, B1);
												v = (d11 * d20 - d01 * d21) * denom;
												w = (d00 * d21 - d01 * d20) * denom;
												u = 1.0f - v - w;
												Vector3<float> frag_coord = triangle.vertices[0] * u + triangle.vertices[1] * v + triangle.vertices[2] * w;
												if (depth_buffer[ix + iy * full_resolution.x] == 0 || frag_coord.z > depth_buffer[ix + iy * full_resolution.x])
												{
													float z_diff = depth_buffer[ix + iy * full_resolution.x] - frag_coord.z;
													if ((z_diff > 0 ? z_diff : -z_diff) < 0.5) break;
													Vector2<float> uv = triangle.uv.x * u + triangle.uv.y * v + triangle.uv.z * w;
													triangle.normal = (triangle.normals.x * uv.x + triangle.normals.y * uv.y + triangle.normals.z * (1.0 - uv.x - uv.y)) * -1;
													
													float lums = max(0.0f, dotf(triangle.normal, Vector3<float>(0.333, 0.333, 0.333)));
													buffer[(int)(ix + iy * full_resolution.x) * 3] = triangle.color.x * lums * u;
													buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = triangle.color.y * lums * v;
													buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = triangle.color.z * lums * w;
													depth_buffer[ix + iy * full_resolution.x] = frag_coord.z;
												}
											}

											CX1 -= FDY12;
											CX2 -= FDY23;
											CX3 -= FDY31;
										}

										CY1 += FDX12;
										CY2 += FDX23;
										CY3 += FDX31;

									}
								}
							}
						}
					}
				}
				*/ 
			}
		}
		void Cpu::render_buffer(Vector2<unsigned int> resolution, Vector2<unsigned int> full_resolution, Camera camera, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, Atlast::RenderingClasses::Triangle3D* dev_triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size, float* dev_boundings, unsigned int boundings_size)
		{

			Atlast::Rendering::Device::render_buffer << <resolution.y, resolution.x >> > (dev_buffer, dev_triangles, triangles_size, dev_textures, dev_lights, lights_size, camera, dev_boundings, boundings_size / 8, transparent, offset, full_resolution);
			cudaDeviceSynchronize();
		}
		void Cpu::rasterize_buffer(Vector3<float> translate, Vector3<float> rotate, Vector2<unsigned int> full_resolution, bool transparent, Vector2<unsigned int> offset, unsigned char* dev_buffer, float* depth_buffer, Atlast::RenderingClasses::Triangle3D *triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* dev_textures, Atlast::RenderingClasses::Light* dev_lights, unsigned int lights_size)
		{
			Device::rasterize_line<<<max(triangles_size >> 8, 1), 256>>>(translate, rotate, full_resolution, dev_buffer, depth_buffer, triangles, triangles_size, dev_textures, dev_lights, lights_size);
			cudaDeviceSynchronize();
		}
		bool Cpu::triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, Triangle3D& triangle, Triangle3D& triangle1, Triangle3D& triangle2)
		{
			Vector3<float> new_point;
			std::vector<Vector3<float>> inside_points;
			std::vector<Vector3<float>> outside_points;
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[0]) < 0) outside_points.push_back(triangle.vertices[0]);
			else inside_points.push_back(triangle.vertices[0]);
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[1]) < 0) outside_points.push_back(triangle.vertices[1]);
			else inside_points.push_back(triangle.vertices[1]);
			if (signed_distance(point_on_plane, plane_normal, triangle.vertices[2]) < 0) outside_points.push_back(triangle.vertices[2]);
			else inside_points.push_back(triangle.vertices[2]);
			if (inside_points.size() == 0)
			{
				return true;
			}

			if (inside_points.size() == 3)
			{
				return false;
			}

			if (inside_points.size() == 1 && outside_points.size() == 2)
			{
				triangle1.vertices[0] = inside_points[0];
				Vector3<float> new_point;
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[0], new_point);
				triangle1.vertices[1] = new_point;
				line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[1], new_point);
				triangle1.vertices[2] = new_point;

				return true;
			}

			if (inside_points.size() == 2 && outside_points.size() == 1)
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
		bool Device::triangle_clip_against_plane(Vector3<float> point_on_plane, Vector3<float> plane_normal, Triangle3D& triangle, Triangle3D& triangle1, Triangle3D& triangle2)
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
		__global__ void Device::clip_triangles(Triangle3D* triangles, unsigned int triangles_size)
		{
			int tri_idx = (threadIdx.x + blockIdx.x * blockDim.x);
			if (tri_idx < triangles_size * 0.5f)
			{
				Triangle3D& triangle = triangles[tri_idx];
				Triangle3D tri1 = triangle;
				Triangle3D tri2 = triangle;

				tri1.vertices[0] = Vector3<float>(0, 0, 1);
				tri1.vertices[1] = Vector3<float>(0, 0, 1);
				tri1.vertices[2] = Vector3<float>(0, 0, 1);
				tri2.vertices[0] = Vector3<float>(0, 0, 1);
				tri2.vertices[1] = Vector3<float>(0, 0, 1);
				tri2.vertices[2] = Vector3<float>(0, 0, 1);
				bool use_new;
				use_new = Atlast::Rendering::Device::triangle_clip_against_plane(Vector3<float>(0, 0, -0.1f), Vector3<float>(0, 0, -1), triangle, tri1, tri2);
				if (use_new)
				{
					triangle = tri1;
					if (tri2.vertices[0].z != 1)
					{
						triangles[(int)(tri_idx + triangles_size * 0.5f) - 1] = tri2;
					}
				}
			}
		}
		bool Cpu::clip_triangles(std::vector<Atlast::RenderingClasses::Triangle3D>& triangles)
		{
			triangles.resize(triangles.size() * 2);
			Triangle3D* dev_triangles = nullptr;
			cudaMalloc(&dev_triangles, sizeof(Triangle3D) * triangles.size());
			cudaMemcpy(dev_triangles, triangles.data(), sizeof(Triangle3D) * triangles.size(), cudaMemcpyHostToDevice);
			Device::clip_triangles<<<256, 512>>>(dev_triangles, triangles.size());
			cudaDeviceSynchronize();
			
			cudaError_t erro = cudaGetLastError();
			if (erro != cudaSuccess)
			{
				std::cout << cudaGetErrorString(erro) << std::endl;
				return false;
			}
			cudaMemcpy(triangles.data(), dev_triangles, sizeof(Triangle3D) * triangles.size(), cudaMemcpyDeviceToHost);
			cudaFree(dev_triangles);
			triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [](const Triangle3D& x) {
				return x.vertices[0].x == 0;
			}));
			return true;
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
		__global__ void HelperFuncs::_rotate(Vector3<float> rotate_by, Vector3<float> center, Triangle3D* triangles, unsigned int triangle_count)
		{
			int i = threadIdx.x;
			int j = blockIdx.x;
			int tid = i + j * 512;
			if (tid < triangle_count)
			{
				Triangle3D& triangle = triangles[tid];
				triangle.vertices[0] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[0], rotate_by, center);
				triangle.vertices[1] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[1], rotate_by, center);
				triangle.vertices[2] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[2], rotate_by, center);
			}
		}
	}
}