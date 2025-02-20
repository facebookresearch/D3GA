// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "tet/generator.h"

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writeMESH.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <cmath>
#include <iostream>
#include <string>

static void print(const Eigen::Vector3d &v)
{
    std::cout << v.x() << " " << v.y() << " " << v.z() << std::endl;
}

static void print(const Eigen::Vector4d &v)
{
    std::cout << v.sum() << " " << v.x() << " " << v.y() << " " << v.z() << " " << v.w() << std::endl;
}

Generator::Generator(std::string input, std::string output, std::string switches, bool generate_bary) : m_input_mesh(input), m_output_mesh(output), m_switches(switches), m_generate_bary(generate_bary)
{
}

void Generator::run()
{
    std::cout << "SWITCHES = " << m_switches << std::endl;

    read_mesh(m_input_mesh);
    tetrahedralize();
    calculate_barycentric();
}

void Generator::tetrahedralize()
{
    for (auto &tuple : m_cages)
    {
        auto mesh = tuple.second;

        Eigen::MatrixXd V = mesh->V;
        Eigen::MatrixXi F = mesh->F;

        Eigen::MatrixXd TV;
        Eigen::MatrixXi TT;
        Eigen::MatrixXi TF;

        igl::copyleft::tetgen::tetrahedralize(V, F, m_switches, TV, TT, TF);
        auto tet = std::make_shared<Tetrahedron>(TV, TT, TF, mesh->get_key());
        m_tetraherdons.push_back(tet);
    }
}

void Generator::calculate_barycentric()
{
    for (const auto &tetra : m_tetraherdons)
    {
        fs::path path = fs::path(m_input_mesh) / m_cages[tetra->key]->actor;

        Eigen::MatrixXd TV = tetra->TV;
        Eigen::MatrixXi TT = tetra->TT;
        Eigen::MatrixXi TF = tetra->TF;

        igl::writeMESH((fs::path(m_output_mesh) / "cage.mesh").string(), TV, TT, TF);

        if (!m_generate_bary) continue;

        // Calculate barycentrics for an object
        auto mesh = m_objects[tetra->key];
        Eigen::MatrixXd V = mesh->V;
        Eigen::MatrixXi F = mesh->F;

        std::cout << "Generating bary for [" << tetra->key << "] with #vertices " << V.rows() << " #triangles " << F.rows() << std::endl;

        std::ofstream file((fs::path(path) / "parametrization.txt").string());
        {
            for (int j = 0; j < V.rows(); ++j)
            {
                Eigen::Vector3d p = V.row(j);

                for (int i = 0; i < TT.rows(); ++i)
                {
                    Eigen::Vector4i tet = TT.row(i);

                    Eigen::Vector3d a = TV.row(tet.x());
                    Eigen::Vector3d b = TV.row(tet.y());
                    Eigen::Vector3d c = TV.row(tet.z());
                    Eigen::Vector3d d = TV.row(tet.w());

                    if (point_in_tet(a, b, c, d, p))
                    {
                        Eigen::Vector4d bary = barycentric(a, b, c, d, p);
                        // assert(int(bary.sum()));
                        // std::cout << bary.sum() << std::endl;
                        file << i << " " << bary.x() << " " << bary.y() << " " << bary.z() << " " << bary.w() << std::endl;
                        break;
                    }
                }
            }
        }
        file.close();
    }
}

void Generator::read_mesh(std::string path)
{   
    fs::path entry = fs::path(path);

    if (!fs::exists(entry)) {
        std::cout << entry.string() << " does not exists!" << std::endl;
    };

    auto cage = get_mesh(entry);
    m_cages[cage->get_key()] = cage;

    if (m_generate_bary) {
        auto object = get_mesh(entry.parent_path() / "object.ply");
        m_objects[object->get_key()] = object;
    }
}

std::shared_ptr<Mesh> Generator::get_mesh(fs::path path)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    if (path.extension().string().find("ply") != std::string::npos)
    {
        igl::readPLY(path.string(), V, F);
    }
    else
    {
        igl::readOBJ(path.string(), V, F);
    }

    std::string actor = path.parent_path().stem().string();
    std::string name = path.stem().string();

    std::cout << "Loaded mesh [ " << path.string() << " ] with #vertices " << V.rows() << " #triangles " << F.rows() << std::endl;

    return std::make_shared<Mesh>(V, F, name, actor);
}
