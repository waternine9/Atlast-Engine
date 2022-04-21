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
		template<class T>
		class Node
		{
		public:
			T value;
			Node<T>* next = NULL;
			__host__ __device__ Node<T>* operator++()
			{
				return next;
			}
		};
		template<class T>
		class LinkedList
		{
		private:
			Node<T>* head = NULL;
			Node<T>* tail = NULL;
		public:
			__host__ __device__ LinkedList(int amount, T fill)
			{
				for (int i = 0;i < amount;i++)
				{
					this->add(fill);
				}
			}
			__host__ __device__ LinkedList()
			{
				// Default
			}
			__host__ __device__ void add(T value)
			{
				Node<T>* temp = new Node<T>;
				temp->value = value;
				if (head == NULL)
				{

					this->head = temp;
					this->tail = temp;
				}
				else
				{
					this->tail->next = temp;
					this->tail = this->tail->next;
				}
			}
			__host__ __device__ void print_float()
			{
				Node<T>* p = this->head;
				while (p != NULL)
				{
					printf("%f ", p->value);
					p = p->next;
				}

				printf("\n");
			}
			__host__ __device__ int size()
			{
				int cur_size = 0;
				Node<T>* p = this->head;
				while (p != NULL)
				{
					cur_size++;
					p = p->next;
				}
				return cur_size;
			}
			__host__ __device__ Node<T>* begin()
			{
				return this->head;
			}
			__host__ __device__ Node<T>* end()
			{
				return this->tail;
			}
			__host__ __device__ Node<T>* next(int idx)
			{
				Node<T>* p = this->head;
				for (int i = 0;i < idx;i++)
				{
					p = p->next;
				}
				return p;
			}
			__host__ __device__ void pop(Node<T>* before)
			{
				Node<T>* temp;
				temp = before->next;
				before->next = temp->next;
				delete temp;

			}
			__host__ __device__ void concat(Node<T>* _begin)
			{
				this->tail->next = _begin;
			}
			__host__ __device__ void fill(T value)
			{
				Node<T>* p = this->head;
				for (int i = 0;i < this->size();i++)
				{
					p->value = value;
					p = p->next;
				}
			}
			__host__ __device__ T& operator[](int idx)
			{
				return this->next(idx)->value;
			}
		};
		class Material
		{
		public:
			int texture_idx = 0;
			bool textured = false;
			bool glass = false;
			bool reflective = false;
			bool smooth_shading = false;
			float reflective_index = 0.5f;
			Vector3<unsigned char> color;
		};
		class Triangle3D
		{
		public:
			unsigned int i0;
			unsigned int i1;
			unsigned int i2;
			unsigned int uv0;
			unsigned int uv1;
			unsigned int uv2;
			unsigned int material_idx;
		};
		class PopulatedTriangle3D
		{
		public:
			Vector3<float> vertices[3];
		};
		namespace HelperFuncs
		{
			int len(std::string str);
			void split(std::string str, char seperator, std::vector<std::string>& substrings);
		}
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
			void push_back_triangle(Triangle3D triangle)
			{
				this->triangles.push_back(triangle);
			}
		};
		class Texture2D
		{
		public:
			unsigned char data[1024][1024][3];
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
			std::vector<float> unique_verts;
			std::vector<float> unique_uv;
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
			bool load_multiple_from_object_file(std::string sFilename, unsigned int material_idx)
			{
				std::ifstream f(sFilename);
				if (!f.is_open())
					return false;

				// Local cache of verts
				std::vector<Vector3<float>> verts;
				std::vector<Vector2<float>> uv;
				int i = -1;

				GameObject cur_game_object;
				bool passed;
				while (!f.eof())
				{
					char line[128];
					f.getline(line, 128);

					std::strstream s;
					s << line;

					char junk;
					if (line[0] == 'o')
					{
						if (passed)
						{
							this->game_objects.push_back(cur_game_object);
						}
						GameObject new_object;

						s >> junk >> new_object.name;
						cur_game_object = new_object;

						passed = true;
					}
					if (line[0] == 'v' && line[1] == ' ')
					{
						Vector3<float> v;
						s >> junk >> v.x >> v.y >> v.z;

						this->unique_verts.push_back(v.x);
						this->unique_verts.push_back(v.y);
						this->unique_verts.push_back(v.z);
					}
					if (line[0] == 'v' && line[1] == 't')
					{
						Vector2<float> v;
						std::string line_str(line);
						std::stringstream check1(line_str);
						std::string bruh;
						std::vector<std::string> tokens;
						while (std::getline(check1, bruh, ' '))
						{
							tokens.push_back(bruh);
						}
						this->unique_uv.push_back(std::stof(tokens[1]));
						this->unique_uv.push_back(1.0f - std::stof(tokens[2]));
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
						triangle.i0 = std::stoi(fa[0]) - 1;
						triangle.i1 = std::stoi(fb[0]) - 1;
						triangle.i2 = std::stoi(fc[0]) - 1;
						triangle.uv0 = std::stoi(fa[1]) - 1;
						triangle.uv1 = std::stoi(fb[1]) - 1;
						triangle.uv2 = std::stoi(fc[1]) - 1;
						triangle.material_idx = material_idx;
						cur_game_object.push_back_triangle(triangle);
					}
				}
				this->game_objects.push_back(cur_game_object);
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