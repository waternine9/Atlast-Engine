#include "functions.cuh"

// functions.cuh


namespace Atlast
{
	namespace Rendering
	{
		void Cpu::texture_fill( Atlast::RenderingClasses::Texture2D& tex, const char* texture_directory, bool normal)
		{
			sf::Image image;
			image.loadFromFile(texture_directory);

			for (int y = 0;y < 1024;y++)
			{
				for (int x = 0;x < 1024;x++)
				{
					sf::Color cur_col = image.getPixel(x, y);
					if (!normal)
					{
						tex.data[((int)(y * 1024) + x) * 3 + 2] = cur_col.r;
						tex.data[((int)(y * 1024) + x) * 3 + 1] = cur_col.g;
						tex.data[((int)(y * 1024) + x) * 3] = cur_col.b;
					}
					else
					{
						tex.normal_data[((int)(y * 1024) + x) * 3] = (((float)cur_col.r) / 255 - 0.5f) * 2.0f;
						tex.normal_data[((int)(y * 1024) + x) * 3 + 1] = (((float)cur_col.g) / 255 - 0.5f) * 2.0f;
						tex.normal_data[((int)(y * 1024) + x) * 3 + 2] = (((float)cur_col.b) / 255 - 0.5f) * 2.0f;
					}
				}
			}
		}
		
		
		__device__ Atlast::Vectors::Vector3<float> tbn_normals(Atlast::Vectors::Vector3<float> orig_normal, Atlast::Vectors::Vector3<float> normal, Atlast::RenderingClasses::PopulatedTriangle3D out_tri)
		{


			Atlast::Vectors::Vector3<float> tt;
			Atlast::Vectors::Vector3<float> b;

			Atlast::Vectors::Vector3<float> edge1 = out_tri.vertices[1] - out_tri.vertices[0];
			Atlast::Vectors::Vector3<float> edge2 = out_tri.vertices[2] - out_tri.vertices[0];

			Atlast::Vectors::Vector2<float> deltaUV1 = out_tri.uv[1] - out_tri.uv[0];
			Atlast::Vectors::Vector2<float> deltaUV2 = out_tri.uv[2] - out_tri.uv[0];

			float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

			tt.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
			tt.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
			tt.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

			b.x = -f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
			b.y = -f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
			b.z = -f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
			tt = Atlast::LinearAlgebra::VectorMath::normalizef(tt - Atlast::Vectors::Vector3<float>(Atlast::LinearAlgebra::VectorMath::dotf(tt, orig_normal)) * orig_normal);
			b = Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::LinearAlgebra::VectorMath::crossf(tt, orig_normal));
			float output7[1][3] = { { 0, 0, 0 } };
			float input2[3][3];
			float input1[1][3];
			input2[0][0] = tt.x;
			input2[0][1] = tt.y;
			input2[0][2] = tt.z;
			input2[1][0] = b.x;
			input2[1][1] = b.y;
			input2[1][2] = b.z;
			input2[2][0] = orig_normal.x;
			input2[2][1] = orig_normal.y;
			input2[2][2] = orig_normal.z;
			input1[0][0] = normal.x;
			input1[0][1] = normal.y;
			input1[0][2] = normal.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output7[_][Y] += input1[_][k] * input2[k][Y];
					}
			return Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::Vectors::Vector3<float>((float)output7[0][0], (float)output7[0][1], (float)output7[0][2]));
		}
		__global__ void Device::rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, Atlast::RenderingClasses::PopulatedTriangle3D triangle, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count)
		{
			int i = threadIdx.x;
			int x1 = triangle.vertices[0].x;
			int x2 = triangle.vertices[1].x;
			int x3 = triangle.vertices[2].x;
					
			int y11 = triangle.vertices[0].y;
			int y21 = triangle.vertices[1].y;
			int y31 = triangle.vertices[2].y;
			Atlast::RenderingClasses::Material material = materials[triangle.material_idx];
			Atlast::Vectors::Vector4<int> bounding(
				min(max(min(min(x1, x2), x3), 0), full_resolution.x - 1),
				min(max(max(max(x1, x2), x3), 0), full_resolution.x - 1),
				min(max(min(min(y11, y21), y31), 0), full_resolution.y - 1),
				min(max(max(max(y11, y21), y31), 0), full_resolution.y - 1)
			);
			/* Atlast::Vectors::Vector2<float> true_half_bounding(
				min(min(triangle.vertices[0].x, triangle.vertices[1].x), triangle.vertices[2].x),
				min(min(triangle.vertices[0].y, triangle.vertices[1].y), triangle.vertices[2].y)
			); */ 
			float interpolation = (((float)i) * 0.00390625f);
			float mintomax = bounding.y - bounding.x;
			bounding.x += interpolation * mintomax;
			bounding.y = bounding.x + 0.00390625f * mintomax;
			if (!((bounding.x < 0 && bounding.y < 0) || (bounding.x >= full_resolution.x && bounding.y >= full_resolution.x) || (bounding.z < 0 && bounding.w < 0) || (bounding.z >= full_resolution.x && bounding.w >= full_resolution.x)))
			{

				/* float ztow = 1.0f / ((float)(bounding.w - bounding.z));
				float xtoy = 1.0f / ((float)(bounding.y - bounding.x));
				float true_x = true_half_bounding.x + interpolation;
				float true_y = true_half_bounding.y; */ 

				for (int iy = bounding.z;iy <= bounding.w;iy++)
				{
					
					if (iy < 0 || iy >= full_resolution.y) continue;
					for (int ix = bounding.x;ix <= bounding.y;ix++)
					{
						
						if (ix < 0 || ix >= full_resolution.x) continue;
						float u, v, w;
						Atlast::Vectors::Vector3<float> B2 = Atlast::Vectors::Vector3<float>(ix, iy, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0);
						
						float d20 = Atlast::LinearAlgebra::VectorMath::dotf(B2, triangle.B0);
						float d21 = Atlast::LinearAlgebra::VectorMath::dotf(B2, triangle.B1);
						v = (triangle.d11 * d20 - triangle.d01 * d21) * triangle.denom;
						w = (triangle.d00 * d21 - triangle.d01 * d20) * triangle.denom;
						u = 1.0f - v - w;
						if (v < -0.001f || w < -0.001f || v + w > 1.001f) continue;
						Atlast::Vectors::Vector3<float> frag_coord = triangle.old_vertices[0] * u + triangle.old_vertices[1] * v + triangle.old_vertices[2] * w;

						Atlast::Vectors::Vector3<float> depth_coord = triangle.vertices[0] * u + triangle.vertices[1] * v + triangle.vertices[2] * w;
						if (depth_buffer[ix + iy * full_resolution.x] == 0 || depth_coord.z > depth_buffer[ix + iy * full_resolution.x])
						{
							if (depth_coord.z > -0.1)
							{
								continue;
							}
							Atlast::Vectors::Vector2<float> uv = triangle.uv[0] * u + triangle.uv[1] * v + triangle.uv[2] * w;
							uv.x = max(min(uv.x, 1.0f), 0.0f);
							uv.y = max(min(uv.y, 1.0f), 0.0f);

							Atlast::Vectors::Vector3<unsigned char> triangle_color = material.color;
							if (material.textured)
							{
								Atlast::Vectors::Vector2<float> scaled = uv * 1023.0f * textures[material.texture_idx].repeating;
								int p_idx = (((int)(scaled.y) % 1024) * 1024.0f + ((int)scaled.x % 1024)) * 3;
								if (p_idx < 1024 * 1024 * 3 - 3)
								{
									if (textures[material.texture_idx].albedo)
									{

										triangle_color = Atlast::Vectors::Vector3<unsigned char>(
											textures[material.texture_idx].data[p_idx],
											textures[material.texture_idx].data[p_idx + 1],
											textures[material.texture_idx].data[p_idx + 2]
											);
									}
									if (textures[material.texture_idx].normal)
									{

										Atlast::Vectors::Vector3<float> map_normal(
											textures[material.texture_idx].normal_data[p_idx + 1],
											textures[material.texture_idx].normal_data[p_idx + 2],
											textures[material.texture_idx].normal_data[p_idx]
										);
										triangle.normal = Rendering::tbn_normals(triangle.normal, Atlast::LinearAlgebra::VectorMath::normalizef(map_normal), triangle);
									}

								}
							}

							Atlast::Vectors::Vector3<float> final_color = Atlast::Vectors::Vector3<float>((float)triangle_color.x, (float)triangle_color.y, (float)triangle_color.z);
							float intensity = 0;
							Atlast::Vectors::Vector3<float> avg_col = Atlast::Vectors::Vector3<float>(0);
							if (!material.unlit)
							{
								Atlast::Vectors::Vector3<float> vert = frag_coord;
								for (int light_idx = 0;light_idx < light_count;light_idx++)
								{
									Atlast::Vectors::Vector3<float> itol = (lights[light_idx].pos - vert);
									float bruh = lights[light_idx].intensity * (max(Atlast::LinearAlgebra::VectorMath::dotf(Atlast::LinearAlgebra::VectorMath::normalizef(itol), (triangle.normal)), 0.0f)) * (Algorithms::fast_isqrt(itol.x * itol.x + itol.y * itol.y + itol.z * itol.z));

									avg_col = avg_col + Atlast::Vectors::Vector3<float>(lights[light_idx].color.x * bruh, lights[light_idx].color.y * bruh, lights[light_idx].color.z * bruh);
									intensity += bruh;
								}
								
							}
							final_color = final_color * Atlast::Vectors::Vector3<float>(intensity) * avg_col;
							

							buffer[(int)(ix + iy * full_resolution.x) * 3] = final_color.x < 255 ? (int)final_color.x : 255;
							buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = final_color.y < 255 ? (int)final_color.y : 255;
							buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = final_color.z < 255 ? (int)final_color.z : 255;
							depth_buffer[ix + iy * full_resolution.x] = depth_coord.z;
							
						}
						// true_x += xtoy;
					}
					// true_y += ztow;
				}
			}
			

			
		}
		__global__ void Device::lazy_rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* buffer, float* depth_buffer, float* unique_verts, float* uv, Atlast::RenderingClasses::Triangle3D* triangles, unsigned int triangles_size, Atlast::RenderingClasses::Texture2D* textures, Atlast::RenderingClasses::Material* materials, Atlast::RenderingClasses::Light* lights, unsigned int light_count)
		{
			int i = threadIdx.x;
			int j = blockIdx.x;
			unsigned int triangle_idx = i + j * 256;
			if (triangle_idx < triangles_size)
			{
				Atlast::RenderingClasses::Triangle3D triangle = triangles[triangle_idx];
				unsigned int vert_idx = triangle.i0 * 3;
				Atlast::Vectors::Vector3<float> tri0(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);
				vert_idx = triangle.i1 * 3;
				Atlast::Vectors::Vector3<float> tri1(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);
				vert_idx = triangle.i2 * 3;
				Atlast::Vectors::Vector3<float> tri2(unique_verts[vert_idx], unique_verts[vert_idx + 1], unique_verts[vert_idx + 2]);
				vert_idx = triangle.uv0 * 2;
				Atlast::Vectors::Vector2<float> uv0(uv[vert_idx], uv[vert_idx + 1]);
				vert_idx = triangle.uv1 * 2;
				Atlast::Vectors::Vector2<float> uv1(uv[vert_idx], uv[vert_idx + 1]);
				vert_idx = triangle.uv2 * 2;
				Atlast::Vectors::Vector2<float> uv2(uv[vert_idx], uv[vert_idx + 1]);

				Atlast::Vectors::Vector3<float> triangle_normal = Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::LinearAlgebra::VectorMath::crossf(tri1 - tri0, tri2 - tri0));

				Atlast::Vectors::Vector3<float> old_tri0 = tri0;
				Atlast::Vectors::Vector3<float> old_tri1 = tri1;
				Atlast::Vectors::Vector3<float> old_tri2 = tri2;

				tri0 = LinearAlgebra::VectorMath::rotatef(tri0 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));
				tri1 = LinearAlgebra::VectorMath::rotatef(tri1 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));
				tri2 = LinearAlgebra::VectorMath::rotatef(tri2 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));
				
				
				
				float inv_v0 = 1.0f / tri0.z;
				float inv_v1 = 1.0f / tri1.z;
				float inv_v2 = 1.0f / tri2.z;

				if (inv_v0 < 7.0f && inv_v1 < 7.0f && inv_v2 < 7.0f)
				{

					tri0.y = (tri0.y * inv_v0);
					tri1.y = (tri1.y * inv_v1);
					tri2.y = (tri2.y * inv_v2);

					tri0.x = (tri0.x * inv_v0);
					tri1.x = (tri1.x * inv_v1);
					tri2.x = (tri2.x * inv_v2);

					tri0.x = tri0.x * (full_resolution.x >> 1) + (full_resolution.x >> 1);
					tri1.x = tri1.x * (full_resolution.x >> 1) + (full_resolution.x >> 1);
					tri2.x = tri2.x * (full_resolution.x >> 1) + (full_resolution.x >> 1);

					tri0.y = tri0.y * (full_resolution.y >> 1) + (full_resolution.y >> 1);
					tri1.y = tri1.y * (full_resolution.y >> 1) + (full_resolution.y >> 1);
					tri2.y = tri2.y * (full_resolution.y >> 1) + (full_resolution.y >> 1);

					tri0.z = tri0.z * (full_resolution.x >> 1);
					tri1.z = tri1.z * (full_resolution.x >> 1);
					tri2.z = tri2.z * (full_resolution.x >> 1);



					float y11 = tri0.y;
					float y21 = tri1.y;
					float y31 = tri2.y;

					float x1 = tri0.x;
					float x2 = tri1.x;
					float x3 = tri2.x;
					Atlast::RenderingClasses::Material material = materials[triangle.material_idx];

					Atlast::Vectors::Vector3<float> B0 = Atlast::Vectors::Vector3<float>(x2, y21, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0),
						B1 = Atlast::Vectors::Vector3<float>(x3, y31, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0);
					float d00 = Atlast::LinearAlgebra::VectorMath::dotf(B0, B0);
					float d01 = Atlast::LinearAlgebra::VectorMath::dotf(B0, B1);
					float d11 = Atlast::LinearAlgebra::VectorMath::dotf(B1, B1);
					float denom = 1.0f / (d00 * d11 - d01 * d01);

					Atlast::RenderingClasses::PopulatedTriangle3D populated_triangle;
					populated_triangle.vertices[0] = tri0;
					populated_triangle.vertices[1] = tri1;
					populated_triangle.vertices[2] = tri2;
					Atlast::RenderingClasses::PopulatedTriangle3D new_tris[2] = { populated_triangle };

					Device::triangle_clip_against_plane(Atlast::Vectors::Vector3<float>(0, 0, -0.1), Atlast::Vectors::Vector3<float>(0, 0, -1), populated_triangle, new_tris[0], new_tris[1]);
					for (int sub_triangle_idx = 0;sub_triangle_idx < 2;sub_triangle_idx++)
					{
						Atlast::RenderingClasses::PopulatedTriangle3D clipped_triangle = new_tris[sub_triangle_idx];
						if (clipped_triangle.vertices[0].z == 0) continue;
						Atlast::Vectors::Vector4<int> bounding(
							min(max(min(min((int)clipped_triangle.vertices[0].x, (int)clipped_triangle.vertices[1].x), (int)clipped_triangle.vertices[2].x), 0), full_resolution.x - 1),
							min(max(max(max((int)clipped_triangle.vertices[0].x, (int)clipped_triangle.vertices[1].x), (int)clipped_triangle.vertices[2].x), 0), full_resolution.x - 1),
							min(max(min(min((int)clipped_triangle.vertices[0].y, (int)clipped_triangle.vertices[1].y), (int)clipped_triangle.vertices[2].y), 0), full_resolution.y - 1),
							min(max(max(max((int)clipped_triangle.vertices[0].y, (int)clipped_triangle.vertices[1].y), (int)clipped_triangle.vertices[2].y), 0), full_resolution.y - 1)
						);
						if (!((bounding.x < 0 && bounding.y < 0) || (bounding.x >= full_resolution.x && bounding.y >= full_resolution.x) || (bounding.z < 0 && bounding.w < 0) || (bounding.z >= full_resolution.x && bounding.w >= full_resolution.x)))
						{

							for (int iy = bounding.z;iy <= bounding.w;iy++)
							{

								if (iy < 0 || iy >= full_resolution.y) continue;
								for (int ix = bounding.x;ix <= bounding.y;ix++)
								{

									if (ix < 0 || ix >= full_resolution.x) continue;
									float u, v, w;
									Atlast::Vectors::Vector3<float> B2 = Atlast::Vectors::Vector3<float>(ix, iy, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0);

									float d20 = Atlast::LinearAlgebra::VectorMath::dotf(B2, B0);
									float d21 = Atlast::LinearAlgebra::VectorMath::dotf(B2, B1);
									v = (d11 * d20 - d01 * d21) * denom;
									w = (d00 * d21 - d01 * d20) * denom;
									u = 1.0f - v - w;
									if (v < -0.001f || w < -0.001f || v + w > 1.001f) continue;
									Atlast::Vectors::Vector3<float> frag_coord = old_tri0 * u + old_tri1 * v + old_tri2 * w;
									Atlast::Vectors::Vector3<float> depth_coord = tri0 * u + tri1 * v + tri2 * w;
									if (depth_buffer[ix + iy * full_resolution.x] == 0 || depth_coord.z > depth_buffer[ix + iy * full_resolution.x])
									{
										if (depth_coord.z > -0.1) continue;
										Atlast::Vectors::Vector2<float> uv = uv0 * u + uv1 * v + uv2 * w;
										uv.x = max(min(uv.x, 1.0f), 0.0f);
										uv.y = max(min(uv.y, 1.0f), 0.0f);

										Atlast::Vectors::Vector3<unsigned char> triangle_color = material.color;
										if (material.textured)
										{
											Atlast::Vectors::Vector2<float> scaled = uv * 1023.0f * textures[material.texture_idx].repeating;
											int p_idx = (((int)(scaled.y) % 1024) * 1024.0f + ((int)scaled.x % 1024)) * 3;
											if (p_idx < 1024 * 1024 * 3 - 3)
											{
												if (textures[material.texture_idx].albedo)
												{

													triangle_color = Atlast::Vectors::Vector3<unsigned char>(
														textures[material.texture_idx].data[p_idx],
														textures[material.texture_idx].data[p_idx + 1],
														textures[material.texture_idx].data[p_idx + 2]
														);
												}
												if (textures[material.texture_idx].normal)
												{

													Atlast::Vectors::Vector3<float> map_normal(
														textures[material.texture_idx].normal_data[p_idx + 1],
														textures[material.texture_idx].normal_data[p_idx + 2],
														textures[material.texture_idx].normal_data[p_idx]
													);
													triangle_normal = Rendering::tbn_normals(triangle_normal, Atlast::LinearAlgebra::VectorMath::normalizef(map_normal), clipped_triangle);
												}

											}
										}

										Atlast::Vectors::Vector3<float> final_color = Atlast::Vectors::Vector3<float>((float)triangle_color.x, (float)triangle_color.y, (float)triangle_color.z);
										float intensity = 0;
										Atlast::Vectors::Vector3<float> avg_col = Atlast::Vectors::Vector3<float>(0);
										if (!material.unlit)
										{
											Atlast::Vectors::Vector3<float> vert = frag_coord;
											for (int light_idx = 0;light_idx < light_count;light_idx++)
											{
												Atlast::Vectors::Vector3<float> itol = (lights[light_idx].pos - vert);
												float bruh = lights[light_idx].intensity * (max(Atlast::LinearAlgebra::VectorMath::dotf(Atlast::LinearAlgebra::VectorMath::normalizef(itol), (triangle_normal)), 0.0f)) * (Algorithms::fast_isqrt(itol.x * itol.x + itol.y * itol.y + itol.z * itol.z));

												avg_col = avg_col + Atlast::Vectors::Vector3<float>(lights[light_idx].color.x * bruh, lights[light_idx].color.y * bruh, lights[light_idx].color.z * bruh);
												intensity += bruh;
											}

										}
										final_color = final_color * Atlast::Vectors::Vector3<float>(intensity) * avg_col;


										buffer[(int)(ix + iy * full_resolution.x) * 3] = final_color.x < 255 ? (int)final_color.x : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 1] = final_color.y < 255 ? (int)final_color.y : 255;
										buffer[(int)(ix + iy * full_resolution.x) * 3 + 2] = final_color.z < 255 ? (int)final_color.z : 255;
										depth_buffer[ix + iy * full_resolution.x] = depth_coord.z;

									}
								}
							}
						}
					}
				}
			}
		}
		void Cpu::rasterize_buffer(Atlast::Vectors::Vector3<float> translate, Atlast::Vectors::Vector3<float> rotate, Atlast::Vectors::Vector2<unsigned int> full_resolution, unsigned char* dev_buffer, float* depth_buffer, Atlast::RenderingClasses::Scene &scene)
		{
			for (Atlast::RenderingClasses::GameObject game_object : scene.game_objects)
			{
				if (game_object.triangles.size() < 200)
				{
					for (Atlast::RenderingClasses::Triangle3D triangle : game_object.triangles)
					{
						unsigned int vert_idx = triangle.i0 * 3;
						Atlast::Vectors::Vector3<float> tri0(scene.unique_verts[vert_idx], scene.unique_verts[vert_idx + 1], scene.unique_verts[vert_idx + 2]);
						vert_idx = triangle.i1 * 3;
						Atlast::Vectors::Vector3<float> tri1(scene.unique_verts[vert_idx], scene.unique_verts[vert_idx + 1], scene.unique_verts[vert_idx + 2]);
						vert_idx = triangle.i2 * 3;
						Atlast::Vectors::Vector3<float> tri2(scene.unique_verts[vert_idx], scene.unique_verts[vert_idx + 1], scene.unique_verts[vert_idx + 2]);

						vert_idx = triangle.uv0 * 2;
						Atlast::Vectors::Vector2<float> uv0(scene.unique_uv[vert_idx], scene.unique_uv[vert_idx + 1]);
						vert_idx = triangle.uv1 * 2;
						Atlast::Vectors::Vector2<float> uv1(scene.unique_uv[vert_idx], scene.unique_uv[vert_idx + 1]);
						vert_idx = triangle.uv2 * 2;
						Atlast::Vectors::Vector2<float> uv2(scene.unique_uv[vert_idx], scene.unique_uv[vert_idx + 1]);

						Atlast::Vectors::Vector3<float> old_tri0 = tri0;
						Atlast::Vectors::Vector3<float> old_tri1 = tri1;
						Atlast::Vectors::Vector3<float> old_tri2 = tri2;

						Atlast::Vectors::Vector3<float> triangle_normal = Atlast::LinearAlgebra::VectorMath::normalizef(Atlast::LinearAlgebra::VectorMath::crossf(tri1 - tri0, tri2 - tri0));

						tri0 = LinearAlgebra::VectorMath::rotatef(tri0 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));
						tri1 = LinearAlgebra::VectorMath::rotatef(tri1 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));
						tri2 = LinearAlgebra::VectorMath::rotatef(tri2 + translate, rotate, Atlast::Vectors::Vector3<float>(0.0f));

						if (tri0.z > -0.1f && tri1.z > -0.1f && tri2.z > -0.1f) continue;

						



						Atlast::RenderingClasses::PopulatedTriangle3D populated_triangle;
						populated_triangle.vertices[0] = tri0;
						populated_triangle.vertices[1] = tri1;
						populated_triangle.vertices[2] = tri2;


						Atlast::RenderingClasses::PopulatedTriangle3D new_tris[2] = { populated_triangle };
						if (tri0.z > -0.1f || tri1.z > -0.1f || tri2.z > -0.1f) Device::triangle_clip_against_plane(Atlast::Vectors::Vector3<float>(0, 0, -0.1), Atlast::Vectors::Vector3<float>(0, 0, -1), populated_triangle, new_tris[0], new_tris[1]);
						for (int sub_triangle_idx = 0;sub_triangle_idx < 2;sub_triangle_idx++)
						{
							Atlast::RenderingClasses::PopulatedTriangle3D clipped_triangle = new_tris[sub_triangle_idx];
							if (clipped_triangle.vertices[0].z == 0.0f) break;
							Atlast::Vectors::Vector3<float> v0 = clipped_triangle.vertices[0];
							Atlast::Vectors::Vector3<float> v1 = clipped_triangle.vertices[1];
							Atlast::Vectors::Vector3<float> v2 = clipped_triangle.vertices[2];

							float inv_v0 = 1.0f / v0.z;
							float inv_v1 = 1.0f / v1.z;
							float inv_v2 = 1.0f / v2.z;

							float y11 = (v0.y * inv_v0);
							float y21 = (v1.y * inv_v1);
							float y31 = (v2.y * inv_v2);

							float x1 = (v0.x * inv_v0);
							float x2 = (v1.x * inv_v1);
							float x3 = (v2.x * inv_v2);
							x1 = x1 * (full_resolution.x >> 1) + (full_resolution.x >> 1);
							x2 = x2 * (full_resolution.x >> 1) + (full_resolution.x >> 1);
							x3 = x3 * (full_resolution.x >> 1) + (full_resolution.x >> 1);

							y11 = y11 * (full_resolution.y >> 1) + (full_resolution.y >> 1);
							y21 = y21 * (full_resolution.y >> 1) + (full_resolution.y >> 1);
							y31 = y31 * (full_resolution.y >> 1) + (full_resolution.y >> 1);
							Atlast::Vectors::Vector3<float> B0 = Atlast::Vectors::Vector3<float>(x2, y21, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0),
								B1 = Atlast::Vectors::Vector3<float>(x3, y31, 0) - Atlast::Vectors::Vector3<float>(x1, y11, 0);
							float d00 = Atlast::LinearAlgebra::VectorMath::dotf(B0, B0);
							float d01 = Atlast::LinearAlgebra::VectorMath::dotf(B0, B1);
							float d11 = Atlast::LinearAlgebra::VectorMath::dotf(B1, B1);
							float denom = 1.0f / (d00 * d11 - d01 * d01);
							clipped_triangle.vertices[0].x = x1;
							clipped_triangle.vertices[0].y = y11;
							clipped_triangle.vertices[0].z = v0.z * (full_resolution.x >> 1);

							clipped_triangle.vertices[1].x = x2;
							clipped_triangle.vertices[1].y = y21;
							clipped_triangle.vertices[1].z = v1.z * (full_resolution.x >> 1);

							clipped_triangle.vertices[2].x = x3;
							clipped_triangle.vertices[2].y = y31;
							clipped_triangle.vertices[2].z = v2.z * (full_resolution.x >> 1);

							clipped_triangle.old_vertices[0] = old_tri0;
							clipped_triangle.old_vertices[1] = old_tri1;
							clipped_triangle.old_vertices[2] = old_tri1;

							clipped_triangle.uv[0] = uv0;
							clipped_triangle.uv[1] = uv1;
							clipped_triangle.uv[2] = uv2;

							clipped_triangle.denom = denom;
							clipped_triangle.B0 = B0;
							clipped_triangle.B1 = B1;
							clipped_triangle.d00 = d00;
							clipped_triangle.d01 = d01;
							clipped_triangle.d11 = d11;
							clipped_triangle.normal = triangle_normal;
							clipped_triangle.material_idx = triangle.material_idx;

							Device::rasterize_buffer << <1, 256 >> > (translate, rotate, full_resolution, dev_buffer, depth_buffer, clipped_triangle, scene.dev_textures, scene.dev_materials, scene.dev_lights, scene.lights.size());

						}
					}
				}
				else
				{
					Atlast::RenderingClasses::Triangle3D* dev_triangles = nullptr;
					cudaMalloc(&dev_triangles, game_object.triangles.size() * sizeof(Atlast::RenderingClasses::Triangle3D));
					cudaMemcpy(dev_triangles, game_object.triangles.data(), game_object.triangles.size() * sizeof(Atlast::RenderingClasses::Triangle3D), cudaMemcpyHostToDevice);
					Device::lazy_rasterize_buffer<<<max((int)game_object.triangles.size() >> 8, 1), 256>>>(translate, rotate, full_resolution, dev_buffer, depth_buffer, scene.dev_verts, scene.dev_uvs, dev_triangles, game_object.triangles.size(), scene.dev_textures, scene.dev_materials, scene.dev_lights, scene.lights.size());
					cudaDeviceSynchronize();
					cudaFree(dev_triangles);
				}
			}
		}
		__host__ __device__ bool Device::triangle_clip_against_plane(Atlast::Vectors::Vector3<float> point_on_plane, Atlast::Vectors::Vector3<float> plane_normal, Atlast::RenderingClasses::PopulatedTriangle3D& triangle, Atlast::RenderingClasses::PopulatedTriangle3D& triangle1, Atlast::RenderingClasses::PopulatedTriangle3D& triangle2)
		{
			Atlast::Vectors::Vector3<float> new_point;
			Atlast::Vectors::Vector3<float> inside_points[4];
			Atlast::Vectors::Vector3<float> outside_points[4];
			int outside_point_count = 0;
			int inside_point_count = 0;
			
			if (Atlast::LinearAlgebra::VectorMath::signed_distance(point_on_plane, plane_normal, triangle.vertices[0]) < 0)
			{
				outside_point_count++;
				outside_points[outside_point_count - 1] = triangle.vertices[0];
			}
			else
			{
				inside_point_count++;
				inside_points[inside_point_count - 1] = triangle.vertices[0];
			}
			if (Atlast::LinearAlgebra::VectorMath::signed_distance(point_on_plane, plane_normal, triangle.vertices[1]) < 0)
			{
				outside_point_count++;
				outside_points[outside_point_count - 1] = triangle.vertices[1];
			}
			else
			{
				inside_point_count++;
				inside_points[inside_point_count - 1] = triangle.vertices[1];
			}
			if (Atlast::LinearAlgebra::VectorMath::signed_distance(point_on_plane, plane_normal, triangle.vertices[2]) < 0)
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
				Atlast::Vectors::Vector3<float> new_point;
				Atlast::LinearAlgebra::VectorMath::line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[0], new_point);
				triangle1.vertices[1] = new_point;
				Atlast::LinearAlgebra::VectorMath::line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[1], new_point);
				triangle1.vertices[2] = new_point;

				return true;
			}

			if (inside_point_count == 2 && outside_point_count == 1)
			{
				triangle1.vertices[0] = inside_points[0];
				triangle1.vertices[1] = inside_points[1];
				Atlast::Vectors::Vector3<float> new_point;
				Atlast::LinearAlgebra::VectorMath::line_clip_against_plane(point_on_plane, plane_normal, inside_points[0], outside_points[0], new_point);
				triangle1.vertices[2] = new_point;

				triangle2.vertices[0] = inside_points[1];
				triangle2.vertices[1] = triangle1.vertices[2];
				Atlast::LinearAlgebra::VectorMath::line_clip_against_plane(point_on_plane, plane_normal, inside_points[1], outside_points[0], new_point);
				triangle2.vertices[2] = new_point;

				return true;
			}
			return false;
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