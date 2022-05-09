#pragma once
#include "../Math/math.cuh"
#include <vector>
#include <string>
#include <strstream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <SFML/Network.hpp>
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
			bool unlit = false;
			bool glass = false;
			bool reflective = false;
			bool smooth_shading = false;
			float reflective_index = 0.5f;
			Atlast::Vectors::Vector3<unsigned char> color;
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
		class ProtoFragment
		{
		public:
			float u;
			float v;
			float w;
			float lum1;
			float lum2;
			float lum3;
			unsigned int i0;
			unsigned int i1;
			unsigned int i2;
			unsigned int material_idx;
		};
		class PopulatedTriangle3D
		{
		public:
			Atlast::Vectors::Vector3<float> vertices[3];
			Atlast::Vectors::Vector3<float> old_vertices[3];
			Atlast::Vectors::Vector2<float> uv[3];
			Atlast::Vectors::Vector3<float> normal;
			Atlast::Vectors::Vector3<float> B0;
			Atlast::Vectors::Vector3<float> B1;
			int material_idx;
			float denom;
			float d00;
			float d01;
			float d11;
		};
		namespace HelperFuncs
		{
			int len(std::string str);
			void split(std::string str, char seperator, std::vector<std::string>& substrings);
		}
		class Camera
		{
		public:
			Atlast::Vectors::Vector3<float> pos;
			Atlast::Vectors::Vector3<float> rot;
			Atlast::Vectors::Vector3<unsigned char> bg_col;
			__host__ __device__ Camera(Atlast::Vectors::Vector3<float> pos, Atlast::Vectors::Vector3<float> rot, Atlast::Vectors::Vector3<unsigned char> bg_col)
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
			Atlast::Vectors::Vector3<float> pos;
			Atlast::Vectors::Vector3<float> color;
			float intensity;
			__host__ __device__ Light(Atlast::Vectors::Vector3<float> pos, Atlast::Vectors::Vector3<float> color, float intensity)
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
			unsigned char data[1024 * 1024 * 3];
			float normal_data[1024 * 1024 * 3];
			bool albedo = false;
			bool normal = false;
			float repeating = 1;
			Atlast::Vectors::Vector2<int> offset;
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
			std::vector<Texture2D> textures;
			std::vector<Material> materials;
			std::vector<float> unique_verts;
			std::vector<float> unique_uv;
			int triangle_count;
			sf::RenderWindow window;
			Triangle3D* dev_triangles = nullptr;
			Material* dev_materials;
			Texture2D* dev_textures = nullptr;
			Light* dev_lights = nullptr;
			float* dev_verts;
			float* dev_uvs;
			

			Scene() 
			{
				this->window.create(sf::VideoMode(512, 512), "Atlast Engine");
			};
			~Scene()
			{
				this->window.close();
				this->free();
			}
			void create_window(unsigned int resolution_x, unsigned int resolution_y, std::string title, bool fullscreen)
			{
				this->window.create(sf::VideoMode(resolution_x, resolution_y), title);
			}
			void poll_events()
			{
				sf::Event event;
				while (this->window.pollEvent(event))
				{
					if (event.type == sf::Event::Closed)
					{
						this->window.close();
					}
				}
			}
			Atlast::Vectors::Vector2<int> get_mouse_pos()
			{
				sf::Vector2i position = sf::Mouse::getPosition() - this->window.getPosition();
				return Atlast::Vectors::Vector2<int>(position.x, position.y);
			}
			void update(unsigned char* buffer)
			{
				
				sf::Texture texture;
				

				sf::Vector2u window_size = this->window.getSize();

				sf::Image render_image;
				render_image.create(window_size.x, window_size.y);

				for (int y = 0;y < window_size.y;y++)
				{
					for (int x = 0;x < window_size.x;x++)
					{
						render_image.setPixel(x, y, sf::Color(buffer[(x + y * window_size.x) * 3], buffer[(x + y * window_size.x) * 3 + 1], buffer[(x + y * window_size.x) * 3 + 2]));
						buffer[(x + y * window_size.x) * 3] = 0;
						buffer[(x + y * window_size.x) * 3 + 1] = 0;
						buffer[(x + y * window_size.x) * 3 + 2] = 0;
					}
				}
				texture.loadFromImage(render_image);
				

				sf::Sprite render_sprite(texture);
				render_sprite.setTextureRect(sf::IntRect(0, 0, window_size.x, window_size.y));
				
				this->window.clear();
				this->window.draw(render_sprite);
				this->window.display();
				
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
			void compile()
			{
				std::vector<Triangle3D> triangles = this->render_setup();
				cudaMalloc(&this->dev_triangles, sizeof(Triangle3D) * triangles.size());
				cudaMemcpy(this->dev_triangles, triangles.data(), sizeof(Triangle3D) * triangles.size(), cudaMemcpyHostToDevice);

				cudaMalloc(&this->dev_materials, sizeof(Material) * materials.size());
				cudaMemcpy(this->dev_materials, materials.data(), sizeof(Material) * materials.size(), cudaMemcpyHostToDevice);


				cudaMalloc(&this->dev_verts, sizeof(float) * this->unique_verts.size());
				cudaMemcpy(this->dev_verts, this->unique_verts.data(), sizeof(float) * this->unique_verts.size(), cudaMemcpyHostToDevice);

				cudaMalloc(&this->dev_uvs, sizeof(float) * this->unique_uv.size());
				cudaMemcpy(this->dev_uvs, this->unique_uv.data(), sizeof(float) * this->unique_uv.size(), cudaMemcpyHostToDevice);


				cudaMalloc(&this->dev_textures, (sizeof(Texture2D)) * this->textures.size());
				cudaMemcpy(this->dev_textures, this->textures.data(), (sizeof(Texture2D)) * this->textures.size(), cudaMemcpyHostToDevice);

				cudaMalloc(&this->dev_lights, sizeof(Light) * this->lights.size());
				cudaMemcpy(this->dev_lights, this->lights.data(), sizeof(Light) * this->lights.size(), cudaMemcpyHostToDevice);
			}
			void free()
			{
				cudaFree(this->dev_triangles);
				cudaFree(this->dev_materials);
				cudaFree(this->dev_verts);
				cudaFree(this->dev_uvs);
				cudaFree(this->dev_textures);
				cudaFree(this->dev_lights);
			}
			bool load_lights_from_object_file(std::string sFilename, Atlast::Vectors::Vector3<float> color, float intensity)
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
						Atlast::Vectors::Vector3<float> v;
						s >> junk >> v.x >> v.y >> v.z;
						this->lights.push_back(Light(v, color, intensity));
					}
				}

				return true;
			}
			bool load_multiple_from_object_file(std::string sFilename, unsigned int material_idx, Atlast::Vectors::Vector3<float> translate)
			{
				std::ifstream f(sFilename);
				if (!f.is_open())
					return false;

				// Local cache of verts
				std::vector<Atlast::Vectors::Vector3<float>> verts;
				std::vector<Atlast::Vectors::Vector2<float>> uv;
				int i = -1;

				GameObject cur_game_object;
				bool passed;
				unsigned int size = this->unique_verts.size() / 3;

				unsigned int uv_size = this->unique_uv.size() / 2;

				
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
						Atlast::Vectors::Vector3<float> v;
						s >> junk >> v.x >> v.y >> v.z;

						this->unique_verts.push_back(v.x + translate.x);
						this->unique_verts.push_back(v.y + translate.y);
						this->unique_verts.push_back(v.z + translate.z);
					}
					if (line[0] == 'v' && line[1] == 't')
					{
						Atlast::Vectors::Vector2<float> v;
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
						triangle.i0 = std::stoi(fa[0]) - 1 + size;
						triangle.i1 = std::stoi(fb[0]) - 1 + size;
						triangle.i2 = std::stoi(fc[0]) - 1 + size;
						triangle.uv0 = std::stoi(fa[1]) - 1 + uv_size;
						triangle.uv1 = std::stoi(fb[1]) - 1 + uv_size;
						triangle.uv2 = std::stoi(fc[1]) - 1 + uv_size;
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
				this->triangle_count = 0;
				for (GameObject& game_object : this->game_objects)
				{
					std::vector<Triangle3D> game_object_triangles = game_object.triangles;
					triangles.insert(triangles.end(), game_object_triangles.begin(), game_object_triangles.end());
					this->triangle_count += game_object_triangles.size();
				}
				return triangles;
			}
		};
		
	}
}