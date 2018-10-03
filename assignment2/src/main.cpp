#include <iostream>
#include <cstring>
#include "utilities/OBJLoader.hpp"
#include "utilities/lodepng.h"
#include "rasteriser.hpp"
#include <cstdlib>
#include <mpi.h>
#include "globals.hpp"

int num_processors;
int cur_rank;
int render_arg;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
  MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);

  std::string input("input/sphere.obj");
	std::string output("output/sphere"+std::to_string(cur_rank)+".png");
	unsigned int width = 1920;
	unsigned int height = 1080;
	unsigned int depth = 3;


  printf("Processors: %d, current rank:%d\n", num_processors, cur_rank);

	for (int i = 1; i < argc; i++) {
		if (i < argc -1) {
			if (std::strcmp("-i", argv[i]) == 0) {
				input = argv[i+1];
			} else if (std::strcmp("-o", argv[i]) == 0) {
				output = argv[i+1];
			} else if (std::strcmp("-w", argv[i]) == 0) {
				width = (unsigned int) std::stoul(argv[i+1]);
			} else if (std::strcmp("-h", argv[i]) == 0) {
				height = (unsigned int) std::stoul(argv[i+1]);
			} else if (std::strcmp("-d", argv[i]) == 0) {
				depth = (int) std::stoul(argv[i+1]);
			}
		}
	}
	std::cout << "Loading '" << input << "' file... " << std::endl;

	std::vector<Mesh> meshs = loadWavefront(input, false);

  uint32_t lenghts[3] = {0,0,0};

  if(cur_rank != 0) {
    for (uint32_t i = 0; i < meshs.size(); i++) {
      Mesh *current = &(meshs.at(i));

      lenghts[0] = current->vertices.size();
      current->vertices.clear();

      lenghts[1] = current->textures.size();
      current->textures.clear();

      lenghts[2] = current->normals.size();
      current->normals.clear();
    }
  }

  MPI_Datatype MPI_float4, MPI_float3;


  int blocklen_3[3] = {1,1,1};
  int blocklen_4[4] = {1,1,1,1};

  MPI_Aint displacements_3[3] = {0,sizeof(float), sizeof(float)*2};
  MPI_Aint displacements_4[4] = {0,sizeof(float), sizeof(float)*2, sizeof(float)*3};

  MPI_Datatype types_3[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype types_4[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};

  MPI_Type_create_struct(3, blocklen_3, displacements_3, types_3, &MPI_float3);
  MPI_Type_create_struct(4, blocklen_4, displacements_4, types_4, &MPI_float4);

  MPI_Type_commit(&MPI_float3);
  MPI_Type_commit(&MPI_float4);

  // Here we broadcast
  for (uint32_t i = 0; i < meshs.size(); i++) {
    Mesh *current_mesh = &(meshs.at(i));
    if(cur_rank != 0) {
      current_mesh->vertices.resize(lenghts[0]);
      current_mesh->textures.resize(lenghts[1]);
      current_mesh->normals.resize(lenghts[2]);
    }
    MPI_Bcast(&(current_mesh->vertices.at(0)), current_mesh->vertices.size(), MPI_float4, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(current_mesh->textures.at(0)), current_mesh->textures.size(), MPI_float3, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(current_mesh->normals.at(0)), current_mesh->normals.size(), MPI_float3, 0, MPI_COMM_WORLD);
  }

	std::vector<unsigned char> frameBuffer = rasterise(meshs, width, height, depth);


  std::cout << "Writing image to '" << output << "'..." << std::endl;

	unsigned error = lodepng::encode(output, frameBuffer, width, height);

	if(error)
	{
		std::cout << "An error occurred while writing the image file: " << error
              << ": " << lodepng_error_text(error) << std::endl;
	}

  MPI_Finalize();
	return 0;
}
