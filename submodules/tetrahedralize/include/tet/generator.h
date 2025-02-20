// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GENERATOR_H
#define GENERATOR_H

#include <utility>
#include <filesystem>
#include <vector>
#include <map>
#include <memory>

#include <tet/tetrahedron.h>

#include <Eigen/Core>

namespace fs = std::filesystem;

class Generator {
    public:
        Generator(std::string input, std::string output, std::string switches, bool generate_bary);
        void run();
    private:
        void make();
        void tetrahedralize();
        void read_mesh(std::string path);
        void calculate_barycentric();
        std::shared_ptr<Mesh> get_mesh(fs::path path);

        bool m_generate_bary;
        std::string m_switches = "";
        std::string m_input_mesh;
        std::string m_output_mesh;
        std::map<std::string, std::shared_ptr<Mesh>> m_cages;
        std::map<std::string, std::shared_ptr<Mesh>> m_objects;
        std::vector<std::shared_ptr<Tetrahedron>> m_tetraherdons;
};

#endif