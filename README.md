# Atlast Engine
 At last, a good engine.

This is a GPU rasterizer capable of rendering up to 4 million triangles at 30 fps.

# Requirements

- CUDA
- OpenCV v4.5.3

# How do I use this?

~~In your Visual Studio 2019 solution, right click on your project and go to `Properties->CUDA C/C++->Additional Include Directories`. In there add the directory to the `src` folder. Then go to `Linker->General->Additional Library Directories` and add the directory to the `Build/x64/` folder. Final step! Go to `Linker/Input/Additional Dependencies` and add `Atlast Engine.lib`. You're done! Now have fun with the engine.~~
For now, just add the files in the `include` folder to your project files.
