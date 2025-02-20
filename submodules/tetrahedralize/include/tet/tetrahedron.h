// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TETRAHEDRON_H
#define TETRAHEDRON_H

#include <Eigen/Core>

#include <cmath>

struct Tetrahedron
{   
    Tetrahedron(const Eigen::MatrixXd &tv, const Eigen::MatrixXi &tt, const Eigen::MatrixXi &tf, const std::string &k) : TV(tv), TT(tt), TF(tf), key(k){};
    Tetrahedron(const Tetrahedron &t): TV(t.TV), TT(t.TT), TF(t.TF), key(t.key){};
    Tetrahedron(Tetrahedron&& t): TV(t.TV), TT(t.TT), TF(t.TF), key(t.key){};

    std::string key;
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TF;
};

struct Mesh
{   
    Mesh(const Eigen::MatrixXd &v, const Eigen::MatrixXi &f, const std::string &n, const std::string &a) : V(v), F(f), name(n), actor(a){};
    Mesh(const Mesh& m): V(m.V), F(m.F), name(m.name), actor(m.actor) {};
    Mesh(Mesh&& m): V(m.V), F(m.F), name(m.name), actor(m.actor) {};
    Mesh& operator=(const Mesh &other)
    {
        V = other.V;
        F = other.F;
        return *this;
    }

    std::string actor;
    std::string name;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    std::string get_key() { return actor; };
};

static bool same_side(
    const Eigen::Vector3d &v1,
    const Eigen::Vector3d &v2,
    const Eigen::Vector3d &v3,
    const Eigen::Vector3d &v4,
    const Eigen::Vector3d &p)
{
    auto normal = (v2 - v1).cross(v3 - v1);
    auto dotV4 = normal.dot(v4 - v1);
    auto dotP = normal.dot(p - v1);
    return std::signbit(dotV4) == std::signbit(dotP);
}

static bool point_in_tet(
    const Eigen::Vector3d &v1,
    const Eigen::Vector3d &v2,
    const Eigen::Vector3d &v3,
    const Eigen::Vector3d &v4,
    const Eigen::Vector3d &p)
{
    return same_side(v1, v2, v3, v4, p) &&
           same_side(v2, v3, v4, v1, p) &&
           same_side(v3, v4, v1, v2, p) &&
           same_side(v4, v1, v2, v3, p);
}

static double scalar_triple_product(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &c)
{
    return a.dot(b.cross(c));
}

static Eigen::Vector4d barycentric(
    const Eigen::Vector3d &a,
    const Eigen::Vector3d &b,
    const Eigen::Vector3d &c,
    const Eigen::Vector3d &d,
    const Eigen::Vector3d &p)
{
    Eigen::Vector3d vap = p - a;
    Eigen::Vector3d vbp = p - b;

    Eigen::Vector3d vab = b - a;
    Eigen::Vector3d vac = c - a;
    Eigen::Vector3d vad = d - a;

    Eigen::Vector3d vbc = c - b;
    Eigen::Vector3d vbd = d - b;

    double va6 = scalar_triple_product(vbp, vbd, vbc);
    double vb6 = scalar_triple_product(vap, vac, vad);
    double vc6 = scalar_triple_product(vap, vad, vab);
    double vd6 = scalar_triple_product(vap, vab, vac);
    double v6 = 1.0 / scalar_triple_product(vab, vac, vad);

    return Eigen::Vector4d(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
}

#endif