
#include "Test.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
    struct Vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct Face {
        std::vector<int> v;
        std::vector<int> vt;
        std::vector<int> vn;
    };

    struct EdgeConstraint {
        int a = 0; // 0-based vertex index
        int b = 0; // 0-based vertex index
        double targetLength = 0.0;
    };

    static int resolveObjIndex(int idx, int count) {
        if (idx > 0) {
            return idx - 1; // OBJ positive index is 1-based
        }
        if (idx < 0) {
            return count + idx; // OBJ negative index is relative to end
        }
        return -1;
    }

    static double distance3D(const Vec3& p0, const Vec3& p1) {
        const double dx = static_cast<double>(p0.x) - static_cast<double>(p1.x);
        const double dy = static_cast<double>(p0.y) - static_cast<double>(p1.y);
        const double dz = static_cast<double>(p0.z) - static_cast<double>(p1.z);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    static void parseFaceToken(const std::string& token, int& v, int& vt, int& vn) {
        v = vt = vn = 0;
        const size_t p1 = token.find('/');
        if (p1 == std::string::npos) {
            v = std::stoi(token);
            return;
        }
        const size_t p2 = token.find('/', p1 + 1);
        v = std::stoi(token.substr(0, p1));
        if (p2 == std::string::npos) {
            const std::string t = token.substr(p1 + 1);
            if (!t.empty()) {
                vt = std::stoi(t);
            }
            return;
        }
        const std::string t = token.substr(p1 + 1, p2 - p1 - 1);
        const std::string n = token.substr(p2 + 1);
        if (!t.empty()) {
            vt = std::stoi(t);
        }
        if (!n.empty()) {
            vn = std::stoi(n);
        }
    }
} // namespace
/*
bool lscm(const std::string& inputObjPath, const std::string& outputObjPath) {
    std::ifstream in(inputObjPath);
    if (!in.is_open()) {
        std::cout << "LSCM: failed to open input " << inputObjPath << '\n';
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<std::string> otherLines;
    std::vector<Face> faces;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("v ", 0) == 0) {
            std::istringstream iss(line);
            char c;
            Vec3 p;
            iss >> c >> p.x >> p.y >> p.z;
            positions.push_back(p);
            continue;
        }
        if (line.rfind("f ", 0) == 0) {
            std::istringstream iss(line);
            char c;
            iss >> c;
            Face face;
            std::string token;
            while (iss >> token) {
                int v = 0, vt = 0, vn = 0;
                parseFaceToken(token, v, vt, vn);
                face.v.push_back(v);
                face.vt.push_back(vt);
                face.vn.push_back(vn);
            }
            if (!face.v.empty()) {
                faces.push_back(std::move(face));
            }
            continue;
        }
        otherLines.push_back(line);
    }

    if (positions.empty()) {
        std::cout << "LSCM: no vertices found in " << inputObjPath << '\n';
        return false;
    }

    // Build unique edge constraints from mesh topology to preserve triangle shape.
    std::vector<EdgeConstraint> constraints;
    constraints.reserve(faces.size() * 3);
    std::unordered_map<std::uint64_t, size_t> edgeMap;
    edgeMap.reserve(faces.size() * 3);
    const int vertexCount = static_cast<int>(positions.size());
    auto addConstraint = [&](int ia, int ib) {
        if (ia < 0 || ib < 0 || ia >= vertexCount || ib >= vertexCount || ia == ib) {
            return;
        }
        int a = ia;
        int b = ib;
        if (a > b) {
            std::swap(a, b);
        }
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
            static_cast<std::uint32_t>(b);
        if (edgeMap.find(key) != edgeMap.end()) {
            return;
        }
        EdgeConstraint c;
        c.a = a;
        c.b = b;
        c.targetLength = distance3D(positions[a], positions[b]);
        edgeMap.emplace(key, constraints.size());
        constraints.push_back(c);
    };

    for (const Face& f : faces) {
        if (f.v.size() < 3) {
            continue;
        }
        std::vector<int> ids;
        ids.reserve(f.v.size());
        for (int objIdx : f.v) {
            ids.push_back(resolveObjIndex(objIdx, vertexCount));
        }
        // Polygon boundary edges.
        for (size_t i = 0; i < ids.size(); ++i) {
            addConstraint(ids[i], ids[(i + 1) % ids.size()]);
        }
        // Triangle fan diagonals to better preserve non-triangle polygons.
        for (size_t i = 1; i + 1 < ids.size(); ++i) {
            addConstraint(ids[0], ids[i + 1]);
        }
    }
    if (constraints.empty()) {
        std::cout << "LSCM: no valid edges found in " << inputObjPath << '\n';
        return false;
    }

    // Initial UV from best planar projection (drop smallest bbox axis).
    float minX = std::numeric_limits<float>::max(), minY = minX, minZ = minX;
    float maxX = std::numeric_limits<float>::lowest(), maxY = maxX, maxZ = maxX;
    for (const Vec3& p : positions) {
        minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
    }
    const float rx = maxX - minX;
    const float ry = maxY - minY;
    const float rz = maxZ - minZ;

    std::vector<double> u(vertexCount, 0.0);
    std::vector<double> v(vertexCount, 0.0);
    for (int i = 0; i < vertexCount; ++i) {
        if (rz <= rx && rz <= ry) { // project XY
            u[i] = static_cast<double>(positions[i].x);
            v[i] = static_cast<double>(positions[i].y);
        }
        else if (ry <= rx && ry <= rz) { // project XZ
            u[i] = static_cast<double>(positions[i].x);
            v[i] = static_cast<double>(positions[i].z);
        }
        else { // project YZ
            u[i] = static_cast<double>(positions[i].y);
            v[i] = static_cast<double>(positions[i].z);
        }
    }

    auto normalizeUV = [&]() {
        double minU = std::numeric_limits<double>::max();
        double minV = std::numeric_limits<double>::max();
        double maxU = std::numeric_limits<double>::lowest();
        double maxV = std::numeric_limits<double>::lowest();
        for (int i = 0; i < vertexCount; ++i) {
            minU = std::min(minU, u[i]); maxU = std::max(maxU, u[i]);
            minV = std::min(minV, v[i]); maxV = std::max(maxV, v[i]);
        }
        const double du = (maxU - minU > 1e-12) ? (maxU - minU) : 1.0;
        const double dv = (maxV - minV > 1e-12) ? (maxV - minV) : 1.0;
        for (int i = 0; i < vertexCount; ++i) {
            u[i] = (u[i] - minU) / du;
            v[i] = (v[i] - minV) / dv;
        }
    };
    normalizeUV();

    // Scale target lengths to current UV scale.
    double avg3D = 0.0;
    double avg2D = 0.0;
    for (const EdgeConstraint& c : constraints) {
        const double du = u[c.a] - u[c.b];
        const double dv = v[c.a] - v[c.b];
        avg3D += c.targetLength;
        avg2D += std::sqrt(du * du + dv * dv);
    }
    avg3D /= static_cast<double>(constraints.size());
    avg2D /= static_cast<double>(constraints.size());
    const double lengthScale = (avg3D > 1e-12) ? (avg2D / avg3D) : 1.0;
    for (EdgeConstraint& c : constraints) {
        c.targetLength *= lengthScale;
    }

    // Fix two anchor vertices to remove translation/rotation/scale ambiguity.
    int anchor0 = 0;
    int anchor1 = 0;
    for (int i = 1; i < vertexCount; ++i) {
        if (u[i] < u[anchor0]) {
            anchor0 = i;
        }
        if (u[i] > u[anchor1]) {
            anchor1 = i;
        }
    }
    if (anchor0 == anchor1 && vertexCount > 1) {
        anchor1 = 1;
    }
    const double anchor0u = u[anchor0];
    const double anchor0v = v[anchor0];
    const double anchor1u = u[anchor1];
    const double anchor1v = v[anchor1];

    // Edge-length preserving optimization (pure C++ gradient descent).
    std::vector<double> gradU(vertexCount, 0.0);
    std::vector<double> gradV(vertexCount, 0.0);
    double step = 0.08;
    for (int iter = 0; iter < 400; ++iter) {
        std::fill(gradU.begin(), gradU.end(), 0.0);
        std::fill(gradV.begin(), gradV.end(), 0.0);

        for (const EdgeConstraint& c : constraints) {
            const double du = u[c.a] - u[c.b];
            const double dv = v[c.a] - v[c.b];
            const double d = std::sqrt(du * du + dv * dv) + 1e-12;
            const double diff = d - c.targetLength;
            const double weight = 1.0 / (c.targetLength + 1e-6);
            const double g = 2.0 * weight * diff / d;
            const double gx = g * du;
            const double gy = g * dv;
            gradU[c.a] += gx; gradV[c.a] += gy;
            gradU[c.b] -= gx; gradV[c.b] -= gy;
        }

        for (int i = 0; i < vertexCount; ++i) {
            if (i == anchor0 || i == anchor1) {
                continue;
            }
            u[i] -= step * gradU[i];
            v[i] -= step * gradV[i];
        }

        // Keep anchors fixed.
        u[anchor0] = anchor0u; v[anchor0] = anchor0v;
        u[anchor1] = anchor1u; v[anchor1] = anchor1v;

        // Dampen step over time.
        step *= 0.995;
    }
    normalizeUV();

    std::ofstream out(outputObjPath, std::ios::trunc);
    if (!out.is_open()) {
        std::cout << "LSCM: failed to open output " << outputObjPath << '\n';
        return false;
    }

    out << "# Generated by pure C++ LSCM fallback\n";
    for (const std::string& l : otherLines) {
        if (l.rfind("vt ", 0) == 0 || l.rfind("vn ", 0) == 0) {
            continue;
        }
        out << l << '\n';
    }
    // Write flattened geometry on XY plane (z = 0).
    for (int i = 0; i < vertexCount; ++i) {
        out << "v " << static_cast<float>(u[i]) << ' ' << static_cast<float>(v[i]) << " 0.0\n";
    }
    for (int i = 0; i < vertexCount; ++i) {
        out << "vt " << static_cast<float>(u[i]) << ' ' << static_cast<float>(v[i]) << '\n';
    }
    for (const Face& f : faces) {
        out << "f";
        for (size_t i = 0; i < f.v.size(); ++i) {
            const int resolved = resolveObjIndex(f.v[i], vertexCount);
            if (resolved < 0) {
                continue;
            }
            const int vi = resolved + 1;
            out << ' ' << vi << '/' << vi;
        }
        out << '\n';
    }
    return true;
}
*/

bool lscm(const std::string& inputObjPath, const std::string& outputObjPath) {
    // Parse OBJ: vertices into positions, faces into faces, everything else into otherLines
    std::ifstream in(inputObjPath);
    if (!in.is_open()) {
        std::cout << "LSCM: failed to open input " << inputObjPath << '\n';
        return false;
    }
    std::vector<Vec3> positions;
    std::vector<Face> faces;
    std::vector<std::string> otherLines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("v ", 0) == 0) {
            std::istringstream ss(line); char c; Vec3 p;
            ss >> c >> p.x >> p.y >> p.z;
            positions.push_back(p);
        }
        else if (line.rfind("f ", 0) == 0) {
            std::istringstream ss(line); char c; ss >> c;
            Face f; std::string tok;
            while (ss >> tok) {
                int v = 0, vt = 0, vn = 0;
                try {
                    parseFaceToken(tok, v, vt, vn);
                }
                catch (...) {
                    continue;
                }
                f.v.push_back(v); f.vt.push_back(vt); f.vn.push_back(vn);
            }
            if (!f.v.empty()) faces.push_back(std::move(f));
        }
        else {
            otherLines.push_back(line);
        }
    }
    if (positions.empty()) {
        std::cout << "LSCM: no vertices found\n"; return false;
    }
    const int N = static_cast<int>(positions.size());
    if (N < 2) {
        std::cout << "LSCM: need at least 2 vertices\n";
        return false;
    }

    // Fan triangulation of polygon faces
    std::vector<std::array<int, 3>> tris;
    for (const Face& f : faces) {
        std::vector<int> ids;
        for (int oi : f.v) ids.push_back(resolveObjIndex(oi, N));
        for (size_t i = 1; i + 1 < ids.size(); ++i) {
            int a = ids[0], b = ids[i], c = ids[i + 1];
            if (a >= 0 && b >= 0 && c >= 0 && a < N && b < N && c < N &&
                a != b && b != c && a != c) {
                tris.push_back({ a, b, c });
            }
        }
    }
    if (tris.empty()) {
        std::cout << "LSCM: no triangles\n"; return false;
    }

    // Pick two anchor vertices: farthest pair (subsampled for speed on large meshes)
    int pin0 = 0, pin1 = (N > 1 ? 1 : 0);
    {
        float maxD2 = 0.0f;
        int step = std::max(1, N / 300);          // sample step ~ every N/300 vertices
        for (int i = 0; i < N; i += step) {
            for (int j = i + 1; j < N; j += step) {
                float dx = positions[i].x - positions[j].x;
                float dy = positions[i].y - positions[j].y;
                float dz = positions[i].z - positions[j].z;
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > maxD2) { maxD2 = d2; pin0 = i; pin1 = j; }
            }
        }
    }
    if (pin0 == pin1) {
        pin1 = (pin0 + 1) % N;
    }

    // Free vertex indexing for reduced linear system
    // x[k]      = u at free vertex k,  k in [0, Nf)
    // x[k + Nf] = v at free vertex k,  k in [0, Nf)
    std::vector<int> freeIdx(N, -1);
    int Nf = 0;
    for (int i = 0; i < N; ++i)
        if (i != pin0 && i != pin1) freeIdx[i] = Nf++;

    const int M = 2 * Nf;           // number of unknowns (u and v for each free vertex)
    const double pin0u = 0.0, pin0v = 0.0, pin1u = 1.0, pin1v = 0.0;

    // Build LSCM least-squares system (two real equations per triangle)
    // Reference: Levy et al., "Least Squares Conformal Maps", SIGGRAPH 2002
    //
    // Per triangle (i,j,k): local 2D coords q[0..2] in the triangle plane
    // c[l] = conj( q[(l+1)%3] - q[(l+2)%3] )
    //
    // Real equation:    sum_l  Re(c_l)*u_l - Im(c_l)*v_l = 0
    // Imag equation:    sum_l  Im(c_l)*u_l + Re(c_l)*v_l = 0

    struct Row {
        std::vector<std::pair<int, double>> entries;
        double rhs = 0.0;
        void add(int col, double val) {
            for (auto& e : entries) if (e.first == col) { e.second += val; return; }
            entries.push_back({ col, val });
        }
    };
    std::vector<Row> rows;
    rows.reserve(tris.size() * 2);

    for (const auto& tri : tris) {
        const Vec3& P0 = positions[tri[0]];
        const Vec3& P1 = positions[tri[1]];
        const Vec3& P2 = positions[tri[2]];

        // Edge vectors from corner P0
        double e1x = P1.x - P0.x, e1y = P1.y - P0.y, e1z = P1.z - P0.z;
        double e2x = P2.x - P0.x, e2y = P2.y - P0.y, e2z = P2.z - P0.z;

        double l1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        if (l1 < 1e-12) continue;

        // Cross product norm |e1 x e2| = 2 * triangle area
        double nx = e1y * e2z - e1z * e2y;
        double ny = e1z * e2x - e1x * e2z;
        double nz = e1x * e2y - e1y * e2x;
        double A2 = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (A2 < 1e-12) continue;

        // Triangle local 2D coordinates in the plane
        double dot_e2_e1 = e2x * e1x + e2y * e1y + e2z * e1z;
        double q[3][2] = {
            {0.0,        0.0    },
            {l1,         0.0    },
            {dot_e2_e1 / l1, A2 / l1}   // A2/l1 = height from P2 onto base e1
        };

        // Weight ~ 1 / sqrt(area): larger triangles contribute more
        double w = 1.0 / std::sqrt(A2 * 0.5 + 1e-12);

        // c[l] = conj( q[(l+1)%3] - q[(l+2)%3] )
        double cRe[3], cIm[3];
        for (int l = 0; l < 3; ++l) {
            int la = (l + 1) % 3, lb = (l + 2) % 3;
            cRe[l] = (q[la][0] - q[lb][0]);
            cIm[l] = -(q[la][1] - q[lb][1]);   // conjugate: flip imaginary part
        }

        const int vIdx[3] = { tri[0], tri[1], tri[2] };

        Row row1, row2;
        for (int l = 0; l < 3; ++l) {
            int v = vIdx[l];
            double wu = w * cRe[l];
            double wv1 = w * (-cIm[l]);   // v coeff in real equation
            double wv2 = w * cRe[l];    // v coeff in imag equation
            double wu2 = w * cIm[l];    // u coeff in imag equation

            if (v == pin0 || v == pin1) {
                double pu = (v == pin0) ? pin0u : pin1u;
                double pv = (v == pin0) ? pin0v : pin1v;
                row1.rhs -= wu * pu + wv1 * pv;
                row2.rhs -= wu2 * pu + wv2 * pv;
            }
            else {
                int k = freeIdx[v];
                row1.add(k, wu);   row1.add(k + Nf, wv1);
                row2.add(k, wu2);   row2.add(k + Nf, wv2);
            }
        }
        rows.push_back(std::move(row1));
        rows.push_back(std::move(row2));
    }

    if (rows.empty()) {
        std::cout << "LSCM: no valid (non-degenerate) triangle equations\n";
        return false;
    }

    // CG on normal equations A^T A x = A^T b
    // matvec: y = A*p, then out = A^T*y
    auto matvec = [&](const std::vector<double>& x, std::vector<double>& out) {
        out.assign(M, 0.0);
        std::vector<double> y(rows.size(), 0.0);
        for (size_t r = 0; r < rows.size(); ++r)
            for (const auto& [col, val] : rows[r].entries)
                y[r] += val * x[col];
        for (size_t r = 0; r < rows.size(); ++r)
            for (const auto& [col, val] : rows[r].entries)
                out[col] += val * y[r];
    };

    // A^T b
    std::vector<double> AtB(M, 0.0);
    for (const auto& row : rows)
        for (const auto& [col, val] : row.entries)
            AtB[col] += val * row.rhs;

    // Conjugate gradient iterations
    std::vector<double> x(M, 0.0);
    std::vector<double> r = AtB;
    std::vector<double> p = r;
    std::vector<double> Ap(M);
    double rsold = 0.0;
    for (double ri : r) rsold += ri * ri;

    const int maxIter = M > 0 ? std::max(M * 3, 2000) : 0;
    for (int iter = 0; iter < maxIter; ++iter) {
        matvec(p, Ap);
        double pAp = 0.0;
        for (int i = 0; i < M; ++i) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < 1e-20) break;
        double alpha = rsold / pAp;
        for (int i = 0; i < M; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        double rsnew = 0.0;
        for (double ri : r) rsnew += ri * ri;
        if (rsnew < 1e-12 * std::max(1.0, rsold)) break;
        if (rsold < 1e-30) break;
        double beta = rsnew / rsold;
        for (int i = 0; i < M; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    // Reassemble per-vertex (u,v) from solution x; sanitize non-finite values
    std::vector<double> u(N, 0.0), v(N, 0.0);
    u[pin0] = pin0u; v[pin0] = pin0v;
    u[pin1] = pin1u; v[pin1] = pin1v;
    for (int i = 0; i < N; ++i) {
        if (freeIdx[i] >= 0) {
            u[i] = x[freeIdx[i]];
            v[i] = x[freeIdx[i] + Nf];
        }
    }
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(u[i]) || !std::isfinite(v[i])) {
            u[i] = 0.0;
            v[i] = 0.0;
        }
    }

    // Normalize to [0,1] with uniform scale
    double minU = 1e18, maxU = -1e18, minV = 1e18, maxV = -1e18;
    for (int i = 0; i < N; ++i) {
        minU = std::min(minU, u[i]); maxU = std::max(maxU, u[i]);
        minV = std::min(minV, v[i]); maxV = std::max(maxV, v[i]);
    }
    double scale = std::max({ maxU - minU, maxV - minV, 1e-12 });
    for (int i = 0; i < N; ++i) { u[i] = (u[i] - minU) / scale; v[i] = (v[i] - minV) / scale; }

    // Write OBJ: header lines, then v = (u,v,0), faces as v/v (no separate vt in this export)
    std::ofstream out(outputObjPath, std::ios::trunc);
    if (!out.is_open()) {
        std::cout << "LSCM: failed to open output " << outputObjPath << '\n';
        return false;
    }
    out << "# Generated by LSCM (Levy et al. 2002)\n";
    for (const std::string& l : otherLines) {
        if (l.rfind("vt ", 0) == 0 || l.rfind("vn ", 0) == 0) continue;
        out << l << '\n';
    }
    // Alternate export (original 3D v + vt lines) ¡ª kept commented for reference
   /* for (int i = 0; i < N; ++i)
        out << "v " << positions[i].x << ' ' << positions[i].y << ' ' << positions[i].z << '\n';
    for (int i = 0; i < N; ++i)
        out << "vt " << static_cast<float>(u[i]) << ' ' << static_cast<float>(v[i]) << '\n';
    for (const Face& f : faces) {
        out << "f";
        for (size_t i = 0; i < f.v.size(); ++i) {
            int r = resolveObjIndex(f.v[i], N);
            if (r < 0) continue;
            out << ' ' << (r + 1) << '/' << (r + 1);
        }
        out << '\n';
    }*/

    for (int i = 0; i < N; ++i)
        out << "v " << static_cast<float>(u[i]) << ' ' << static_cast<float>(v[i]) << ' ' << 0.0f << '\n';

    for (const Face& f : faces) {
        out << "f";
        for (size_t i = 0; i < f.v.size(); ++i) {
            int r = resolveObjIndex(f.v[i], N);
            if (r < 0) continue;
            out << ' ' << (r + 1) << '/' << (r + 1);
        }
        out << '\n';
    }
    return true;
}

void lscm() {
    const std::string inputPath = "models/qian.OBJ";
    const std::string outputPath = "models/qian_NL.obj";
    std::cout << "lscm_CPU() start\n";
    std::cout << "input data folder: " << inputPath << '\n';
    std::cout << "output: " << outputPath << '\n';
    const bool ok = lscm(inputPath, outputPath);
    std::cout << (ok ? "lscm() done\n" : "lscm() failed\n");
}