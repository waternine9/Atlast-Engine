#include "functions.cuh"
void Atlast::Rendering::Cpu::texture_fill(Texture2D& tex, const char* texture_directory, bool normal)
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
__global__ void Atlast::Rendering::Device::render_buffer(unsigned char* buffer, Triangle3D* triangles, Camera camera)
{

}
int Atlast::RenderingClasses::HelperFuncs::len(std::string str)
{
	int length = 0;
	for (int i = 0; str[i] != '\0'; i++)
	{
		length++;
	}
	return length;

}
void Atlast::RenderingClasses::HelperFuncs::split(std::string str, char seperator, std::vector<std::string>& substrings)
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