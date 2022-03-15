#include "atlast.cuh"
#include <Windows.h>
using namespace Atlast::RenderingClasses;

int main()
{
	Camera camera;
	Scene scene(camera);
	GameObject object;
	object.load_from_object_file("C:/Users/arthu/ObjFiles/berries.obj", Vector3<unsigned char>(0, 255, 0));
	scene.game_objects.push_back(object);
	
	while (true)
	{
		
		unsigned char* buffer = new unsigned char[512 * 512 * 3];

		std::vector<Triangle3D> triangles = scene.render_setup();
		
		Rendering::Cpu::render_buffer(buffer, triangles, scene.textures, scene.lights, scene.camera, Vector2<unsigned int>(512, 512), true);
		Rendering::Cpu::imshow("Render", buffer, Vector2<unsigned int>(512, 512));
		delete buffer;
		if (GetKeyState('W') & 0x8000)
		{
			camera.rot.x += 0.1f;
		}
		if (GetKeyState('S') & 0x8000)
		{
			camera.rot.x -= 0.1f;
		}
		if (GetKeyState('A') & 0x8000)
		{
			camera.rot.y += 0.1f;
		}
		if (GetKeyState('D') & 0x8000)
		{
			camera.rot.y -= 0.1f;
		}
		if (GetKeyState('R') & 0x8000)
		{
			camera.pos.x += 0.1f;
		}
		if (GetKeyState('T') & 0x8000)
		{
			camera.pos.x -= 0.1f;
		}
		scene.camera = camera;
	}
	
	return 0;
}