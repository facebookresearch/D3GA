// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "tet/generator.h"

#include <iostream>

bool to_bool(std::string ip) {
   bool op;
   std::istringstream(ip) >> std::boolalpha >> op;
   return op;
}

int main(int argc, char *argv[])
{
    // std::string switches = "pq1.414a0.01";
    // std::string switches = "p"; // no extra vertices
    if (argc <= 1) {
        std::cout << "parameters [switches (string)] [input(string)] [output(string)] [generate barycentrics (true/false)]]" << std::endl;
        std::cout << "For instance ./tetra pq1.414a0.01 ../input/cage.ply ../output/cage.mesh false" << std::endl;
        std::cout << "Provide switches for TETGEN! Example: pq1.414a0.01" << std::endl;
        std::cout << "Provide input path and output path. Example" << std::endl;
        std::cout << "  Input:  /home/wzielonka/dataset/actors/" << std::endl;
        std::cout << "  Output: /home/wzielonka/dataset/actors/" << std::endl;
        exit(0);
    }
    std::string switches = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    bool generate_bary = to_bool(argv[4]);

    Generator generator(input, output, switches, generate_bary);

    generator.run();
}

/*
cmake --debug-output -DFETCHCONTENT_SOURCE_DIR_LIBIGL=/mnt/home/wzielonka/projects/libigl  -DFETCHCONTENT_SOURCE_DIR_EIGEN=/mnt/home/wzielonka/projects/eigen-3.4.0  -DFETCHCONTENT_SOURCE_DIR_GLFW=/mnt/home/wzielonka/projects/glfw  -DFETCHCONTENT_SOURCE_DIR_GLAD=/mnt/home/wzielonka/projects/glad  -DFETCHCONTENT_SOURCE_DIR_TETGEN=/mnt/home/wzielonka/projects/tetgen ..
*/

// ./tetra pq1.414a0.01 /mnt/home/wzielonka/dataset/gaussians/ /mnt/home/wzielonka/dataset/gaussians/ false