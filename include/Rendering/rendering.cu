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
		__global__ void Device::render_buffer(unsigned char* buffer, Triangle3D* triangles, unsigned int triangles_count, Texture2D* textures, Light* lights, unsigned int light_count, Camera camera, bool transparent)
		{
			float i = threadIdx.x;
			float j = blockIdx.x;
			float resolution_x = blockDim.x;
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
				camera.rot,
				camera.pos
			);
			Triangle3D cur_tri;
			float closest_t = 1e+10f;
			for (unsigned int tri_idx = 0;tri_idx < triangles_count;tri_idx++)
			{
				float t, u, v;
				if (Atlast::Algorithms::ray_triangle(ray_pos, ray_vec, triangles[tri_idx].vertices[0], triangles[tri_idx].vertices[1], triangles[tri_idx].vertices[2], t, u, v))
				{
					if (t < closest_t)
					{
						closest_t = t;
						cur_tri = triangles[tri_idx];
					}
					
				}
			}
			if (closest_t < 1e+10f)
			{
				buffer[r_idx] = cur_tri.color.x;
				buffer[g_idx] = cur_tri.color.y;
				buffer[b_idx] = cur_tri.color.z;
			}
		}
		void Cpu::render_buffer(unsigned char* buffer, std::vector<Triangle3D> triangles, std::vector<Texture2D> textures, std::vector<Light> lights, Camera camera, Vector2<unsigned int> resolution, bool transparent)
		{
			unsigned char* dev_buffer = nullptr;
			cudaMalloc(&dev_buffer, 3 * resolution.x * resolution.y);
			cudaMemcpy(dev_buffer, buffer, 3 * resolution.x * resolution.y, cudaMemcpyHostToDevice);

			Triangle3D* dev_triangles = nullptr;
			cudaMalloc(&dev_triangles, sizeof(Triangle3D) * triangles.size());
			cudaMemcpy(dev_triangles, triangles.data(), sizeof(Triangle3D) * triangles.size(), cudaMemcpyHostToDevice);

			Texture2D* dev_textures = nullptr;
			cudaMalloc(&dev_textures, sizeof(Texture2D) * textures.size());
			cudaMemcpy(dev_textures, textures.data(), sizeof(Texture2D) * textures.size(), cudaMemcpyHostToDevice);

			Light* dev_lights = nullptr;
			cudaMalloc(&dev_lights, sizeof(lights) * lights.size());
			cudaMemcpy(dev_lights, lights.data(), sizeof(lights) * lights.size(), cudaMemcpyHostToDevice);

			Atlast::Rendering::Device::render_buffer<<<resolution.y, resolution.x>>>(dev_buffer, dev_triangles, triangles.size(), dev_textures, dev_lights, lights.size(), camera, transparent);
			cudaDeviceSynchronize();

			cudaMemcpy(buffer, dev_buffer, 3 * resolution.x * resolution.y, cudaMemcpyDeviceToHost);

			cudaFree(dev_buffer);
			cudaFree(dev_triangles);
			cudaFree(dev_textures);
			cudaFree(dev_lights);
		}

		void Cpu::imshow(const char* window_name, unsigned char* buffer, Vector2<unsigned int> resolution)
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
				}
			}
			cv::imshow(window_name, canvas);
			cv::waitKey(1);
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