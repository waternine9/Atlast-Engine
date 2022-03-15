#pragma once
#include "../Math/math.cuh"
#include <vector>
#include <string>
#include <strstream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
namespace Atlast
{
	namespace RenderingClasses
	{
		namespace HelperFuncs
		{
			int len(std::string str);
			void split(std::string str, char seperator, std::vector<std::string>& substrings);
		}
		class Triangle3D
		{
		public:
			Vector3<float> vertices[3];
			Vector3<Vector2<float>> uv;
			Vector3<Vector3<float>> normals;
			Vector3<unsigned char> color;
			Vector3<float> normal;
			Vector4<int> bounding;
			int texture_idx = 0;
			bool textured = false;
			bool glass = false;
			bool reflective = false;
			bool smooth_shading = false;
			float reflective_index = 0.5f;
			__host__ __device__ Triangle3D()
			{

			}
			__host__ __device__ Triangle3D(Vector3<float> v0, Vector3<float> v1, Vector3<float> v2)
			{
				this->vertices[0] = v0;
				this->vertices[1] = v1;
				this->vertices[2] = v2;
			}
			__host__ __device__ void calc_normal()
			{
				this->normal = normalizef(crossf(this->vertices[2] - this->vertices[0], this->vertices[1] - this->vertices[0]));
			}
		};

		class Camera
		{
		public:
			Vector3<float> pos;
			Vector3<float> rot;
			Vector3<unsigned char> bg_col;
			__host__ __device__ Camera(Vector3<float> pos, Vector3<float> rot, Vector3<unsigned char> bg_col)
			{
				this->pos = pos;
				this->rot = rot;
				this->bg_col = bg_col;
			}
			__host__ __device__ Camera()
			{

			}
		};
		class Light
		{
		public:
			Vector3<float> pos;
			Vector3<unsigned char> color;
			float intensity;
			__host__ __device__ Light(Vector3<float> pos, Vector3<unsigned char> color, float intensity)
			{
				this->pos = pos;
				this->color = color;
				this->intensity = intensity;
			}
			__host__ __device__ Light()
			{

			}
		};
		class GameObject
		{
		public:
			std::vector<Triangle3D> triangles;
			std::string name;
			Vector2<Vector3<float>> bounding;
			bool load_from_object_file(std::string filename, Vector3<unsigned char> color, bool glass = false, bool reflective = false, float reflective_index = 0.0f, int texture_idx = 0, bool textured = false, bool smooth_shading = false)
			{
				std::ifstream f(filename);
				if (!f.is_open())
					return false;

				// Local cache of verts
				std::vector<Vector3<float>> verts;
				std::vector<Vector2<float>> uv;
				std::vector<Vector3<float>> normals;
				int i = -1;
				while (!f.eof())
				{
					char line[128];
					f.getline(line, 128);

					std::strstream s;
					s << line;

					char junk;
					std::string junk2;

					if (line[0] == 'v')
					{
						Vector3<float> v(0.0f);
						s >> junk >> v.x >> v.y >> v.z;

						verts.push_back(v);
					}
					if (line[0] == 'v' && line[1] == 't')
					{
						Vector2<float> v(0.0f);
						std::string line_str(line);
						std::stringstream check1(line_str);
						std::string bruh;
						std::vector<std::string> tokens;
						while (std::getline(check1, bruh, ' '))
						{
							tokens.push_back(bruh);
						}
						v.x = std::stof(tokens[1]);
						v.y = 1.0f - std::stof(tokens[2]);
						uv.push_back(v);
					}
					if (line[0] == 'v' && line[1] == 'n')
					{
						Vector3<float> v(0.0f);
						std::string line_str(line);
						std::stringstream check1(line_str);
						std::string bruh;
						std::vector<std::string> tokens;
						while (std::getline(check1, bruh, ' '))
						{
							tokens.push_back(bruh);
						}
						v.x = std::stof(tokens[1]);
						v.y = std::stof(tokens[2]);
						v.z = std::stof(tokens[3]);
						normals.push_back(v);
					}
					if (line[0] == 'f')
					{
						i++;
						std::vector<std::string> fa;
						std::vector<std::string> fb;
						std::vector<std::string> fc;
						std::string f2;
						std::string f3;
						std::string f4;

						s >> junk >> f2 >> f3 >> f4;
						HelperFuncs::split(f2, '/', fa);
						HelperFuncs::split(f3, '/', fb);
						HelperFuncs::split(f4, '/', fc);


						Triangle3D triangle;
						triangle.vertices[0] = verts[std::stoi(fa[0]) - 1];
						triangle.vertices[1] = verts[std::stoi(fb[0]) - 1];
						triangle.vertices[2] = verts[std::stoi(fc[0]) - 1];
						triangle.uv = Vector3<Vector2<float>>(
							uv[std::stoi(fa[1]) - 1],
							uv[std::stoi(fb[1]) - 1],
							uv[std::stoi(fc[1]) - 1]
							);

						triangle.normals = Vector3<Vector3<float>>(
							normals[std::stoi(fa[2]) - 1],
							normals[std::stoi(fb[2]) - 1],
							normals[std::stoi(fc[2]) - 1]
							);
						triangle.color = color;
						triangle.calc_normal();
						triangle.textured = textured;
						triangle.texture_idx = texture_idx;
						triangle.glass = glass;
						triangle.reflective = reflective;
						triangle.reflective_index = reflective_index;

						triangle.smooth_shading = smooth_shading;
						this->triangles.push_back(triangle);

					}
				}

				return true;
			}
			void translate(Vector3<float> translate_by)
			{
				for (Triangle3D& triangle : this->triangles)
				{
					triangle.vertices[0] = triangle.vertices[0] + translate_by;
					triangle.vertices[1] = triangle.vertices[1] + translate_by;
					triangle.vertices[2] = triangle.vertices[2] + translate_by;
				}
				Vector3<float> min_point = this->bounding.x + translate_by;
				Vector3<float> max_point = this->bounding.y + translate_by;
				Vector3<float> bruh_min_point = Vector3<float>(min(min_point.x, max_point.x), min(min_point.y, max_point.y), min(min_point.z, max_point.z));
				Vector3<float> bruh_max_point = Vector3<float>(max(max_point.x, min_point.x), max(max_point.y, min_point.y), max(max_point.z, min_point.z));
				this->bounding.x = bruh_min_point;
				this->bounding.y = bruh_max_point;
			}
			void push_back_triangle(Triangle3D triangle)
			{
				this->triangles.push_back(triangle);
			}
			void rotate(Vector3<float> rotate_by, Vector3<float> center)
			{
				for (Triangle3D& triangle : this->triangles)
				{
					triangle.vertices[0] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[0], rotate_by, center);
					triangle.vertices[1] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[1], rotate_by, center);
					triangle.vertices[2] = Atlast::LinearAlgebra::VectorMath::rotatef(triangle.vertices[2], rotate_by, center);
					triangle.calc_normal();
				}
				Vector3<float> min_point = Atlast::LinearAlgebra::VectorMath::rotatef(this->bounding.x, rotate_by, center);
				Vector3<float> max_point = Atlast::LinearAlgebra::VectorMath::rotatef(this->bounding.y, rotate_by, center);
				Vector3<float> bruh_min_point = Vector3<float>(min(min_point.x, max_point.x), min(min_point.y, max_point.y), min(min_point.z, max_point.z));
				Vector3<float> bruh_max_point = Vector3<float>(max(max_point.x, min_point.x), max(max_point.y, min_point.y), max(max_point.z, min_point.z));
				this->bounding.x = bruh_min_point;
				this->bounding.y = bruh_max_point;
			}
			void calculate_aabb()
			{
				Vector3<float> min_point(1000000000.0f, 1000000000.0f, 1000000000.0f);
				Vector3<float> max_point(-1000000000.0f, -1000000000.0f, -1000000000.0f);
				for (Triangle3D triangle : this->triangles)
				{
					min_point = Vector3<float>(min(min_point.x, triangle.vertices[0].x), min(min_point.y, triangle.vertices[0].y), min(min_point.z, triangle.vertices[0].z));
					max_point = Vector3<float>(max(max_point.x, triangle.vertices[0].x), max(max_point.y, triangle.vertices[0].y), max(max_point.z, triangle.vertices[0].z));
					min_point = Vector3<float>(min(min_point.x, triangle.vertices[1].x), min(min_point.y, triangle.vertices[1].y), min(min_point.z, triangle.vertices[1].z));
					max_point = Vector3<float>(max(max_point.x, triangle.vertices[1].x), max(max_point.y, triangle.vertices[1].y), max(max_point.z, triangle.vertices[1].z));
					min_point = Vector3<float>(min(min_point.x, triangle.vertices[2].x), min(min_point.y, triangle.vertices[2].y), min(min_point.z, triangle.vertices[2].z));
					max_point = Vector3<float>(max(max_point.x, triangle.vertices[2].x), max(max_point.y, triangle.vertices[2].y), max(max_point.z, triangle.vertices[2].z));
				}
				Vector3<float> bruh_min_point = Vector3<float>(min(min_point.x, max_point.x), min(min_point.y, max_point.y), min(min_point.z, max_point.z));
				Vector3<float> bruh_max_point = Vector3<float>(max(max_point.x, min_point.x), max(max_point.y, min_point.y), max(max_point.z, min_point.z));
				this->bounding.x = bruh_min_point;
				this->bounding.y = bruh_max_point;
			}
		};
		class Texture2D
		{
		public:
			float data[1024][1024][3];
			float normal_data[1024][1024][3];
			bool albedo;
			bool normal;
			float repeating = 1;
			Vector2<int> offset;
			__host__ __device__ Texture2D(bool albedo, bool normal)
			{
				this->normal = normal;
				this->albedo = albedo;
			}
			__host__ __device__ Texture2D()
			{
				// Default
			}
		};
		class Scene
		{

		public:
			std::vector<GameObject> game_objects;
			std::vector<Light> lights;
			std::vector<float> boundary_data;
			Camera camera;
			std::vector<Texture2D> textures;
			Scene(Camera camera)
			{
				this->camera = camera;
			}
			void add_light(Light light)
			{
				this->lights.push_back(light);
			}
			std::vector<Light> get_lights()
			{
				return this->lights;
			}
			void set_light(const int idx, const Light new_light)
			{
				this->lights[idx] = new_light;
			}
			bool load_lights_from_object_file(std::string sFilename, Vector3<unsigned char> color, float intensity)
			{
				std::ifstream f(sFilename);
				if (!f.is_open())
					return false;
				while (!f.eof())
				{
					char line[128];
					f.getline(line, 128);

					std::strstream s;
					s << line;

					char junk;

					if (line[0] == 'v')
					{
						Vector3<float> v;
						s >> junk >> v.x >> v.y >> v.z;
						this->lights.push_back(Light(v, color, intensity));
					}
				}

				return true;
			}
			void load_gameobjects_from_vector(std::vector<GameObject> game_objects)
			{
				for (GameObject game_object : game_objects)
				{
					this->game_objects.push_back(game_object);
				}
			}
			int find_gameobject_by_name(std::string name)
			{
				for (int i = 0;i < this->game_objects.size();i++)
				{
					if (this->game_objects[i].name == name) return i;
				}
				return -1;
			}
			void bake_boundary_data()
			{
				std::vector<float> bounding_data;
				int triangle_counter = 0;
				for (GameObject& game_object : this->game_objects)
				{
					std::vector<Triangle3D> game_object_triangles = game_object.triangles;
					bounding_data.push_back(game_object.bounding.x.x);
					bounding_data.push_back(game_object.bounding.x.y);
					bounding_data.push_back(game_object.bounding.x.z);
					bounding_data.push_back(game_object.bounding.y.x);
					bounding_data.push_back(game_object.bounding.y.y);
					bounding_data.push_back(game_object.bounding.y.z);
					bounding_data.push_back((float)triangle_counter);
					bounding_data.push_back((float)triangle_counter + game_object_triangles.size() - 1);
					triangle_counter += game_object_triangles.size();

				}
				this->boundary_data = bounding_data;
			}
			std::vector<Triangle3D> render_setup()
			{
				std::vector<Triangle3D> triangles;
				for (GameObject& game_object : this->game_objects)
				{
					std::vector<Triangle3D> game_object_triangles = game_object.triangles;
					triangles.insert(triangles.end(), game_object_triangles.begin(), game_object_triangles.end());
				}
				return triangles;
			}
		};
	}
}