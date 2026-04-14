#include "makehuman.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace makehuman {
namespace {

constexpr const char* kGitMakeHumanRoot =
    "/home/roots/develop/resources/makehuman/makehuman/makehuman";
constexpr const char* kLegacyInstallerRoot =
    "/home/roots/develop/resources/makehuman/makehuman-0.9.1/makehuman-0.9.1";

/**
 * Lambert + second fill light in **model space** (fixed vs mesh). Baked into vertex color so the
 * body reads 3D even when the Vulkan fragment shader is a passthrough (common with stale .spv).
 */
glm::vec3 bakeVertexShading(const glm::vec3& baseRgb, glm::vec3 unitNormal) {
    const float len = glm::length(unitNormal);
    if (len > 1e-20f) {
        unitNormal /= len;
    }
    else {
        unitNormal = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    const glm::vec3 L = glm::normalize(glm::vec3(0.38f, 0.82f, 0.42f));
    const float ndl = glm::max(glm::dot(unitNormal, L), 0.0f);
    const glm::vec3 L2 = glm::normalize(glm::vec3(-0.62f, 0.28f, 0.73f));
    const float fill = glm::max(glm::dot(unitNormal, L2), 0.0f);
    constexpr float ambient = 0.24f;
    float k = ambient + 0.66f * ndl + 0.32f * fill;
    k = glm::clamp(k, 0.34f, 1.36f);
    return glm::clamp(baseRgb * k, glm::vec3(0.0f), glm::vec3(1.0f));
}

void pushTri(std::vector<MeshVertex>& out, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& color) {
    glm::vec3 n = glm::cross(b - a, c - a);
    const float nl = glm::length(n);
    n = nl > 1e-20f ? n / nl : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 lit = bakeVertexShading(color, n);
    out.push_back({ a, n, lit });
    out.push_back({ b, n, lit });
    out.push_back({ c, n, lit });
}

void appendUvSphere(std::vector<MeshVertex>& out, const glm::mat4& world, float radius, const glm::vec3& color, int lat, int lon) {
    for (int i = 0; i < lat; ++i) {
        const float v0 = static_cast<float>(i) / static_cast<float>(lat);
        const float v1 = static_cast<float>(i + 1) / static_cast<float>(lat);
        const float phi0 = (v0 - 0.5f) * glm::pi<float>();
        const float phi1 = (v1 - 0.5f) * glm::pi<float>();
        for (int j = 0; j < lon; ++j) {
            const float u0 = static_cast<float>(j) / static_cast<float>(lon);
            const float u1 = static_cast<float>(j + 1) / static_cast<float>(lon);
            const float th0 = u0 * glm::two_pi<float>();
            const float th1 = u1 * glm::two_pi<float>();
            auto p = [&](float phi, float th) {
                const glm::vec4 local(
                    radius * std::cos(phi) * std::cos(th),
                    radius * std::sin(phi),
                    radius * std::cos(phi) * std::sin(th),
                    1.0f);
                return glm::vec3(world * local);
            };
            const glm::vec3 a = p(phi0, th0);
            const glm::vec3 b = p(phi1, th0);
            const glm::vec3 c = p(phi1, th1);
            const glm::vec3 d = p(phi0, th1);
            pushTri(out, a, b, c, color);
            pushTri(out, a, c, d, color);
        }
    }
}

void appendYCylinder(std::vector<MeshVertex>& out, const glm::vec3& center, float radius, float height, const glm::vec3& color, int slices, int stacks) {
    const float half = height * 0.5f;
    for (int j = 0; j < slices; ++j) {
        const float t0 = static_cast<float>(j) / static_cast<float>(slices);
        const float t1 = static_cast<float>(j + 1) / static_cast<float>(slices);
        const float a0 = t0 * glm::two_pi<float>();
        const float a1 = t1 * glm::two_pi<float>();
        for (int i = 0; i < stacks; ++i) {
            const float f0 = static_cast<float>(i) / static_cast<float>(stacks);
            const float f1 = static_cast<float>(i + 1) / static_cast<float>(stacks);
            const float y0 = center.y + glm::mix(-half, half, f0);
            const float y1 = center.y + glm::mix(-half, half, f1);
            const glm::vec3 p00(center.x + radius * std::cos(a0), y0, center.z + radius * std::sin(a0));
            const glm::vec3 p01(center.x + radius * std::cos(a0), y1, center.z + radius * std::sin(a0));
            const glm::vec3 p10(center.x + radius * std::cos(a1), y0, center.z + radius * std::sin(a1));
            const glm::vec3 p11(center.x + radius * std::cos(a1), y1, center.z + radius * std::sin(a1));
            pushTri(out, p00, p10, p11, color);
            pushTri(out, p00, p11, p01, color);
        }
    }
}

/**
 * MakeHuman `base.obj` packs the visible **body** (vertex indices 0..13379) first, then **helper**
 * geometry used inside MH for clothes, eyelashes, joints, etc. (see community basemesh docs).
 * Drawing helpers with the same skin material looks like random limbs or **tights / skirt shells**.
 * When the file looks like a full canonical basemesh, keep only triangles that reference body verts.
 */
constexpr int kMakeHumanCanonicalLastBodyVertexInclusive = 13379;

void filterToCanonicalBodyTriangles(std::vector<std::array<int, 3>>& triangles, size_t positionCount) {
    constexpr size_t kMinPositionsForHelpersPresent = 14000;
    if (positionCount < kMinPositionsForHelpersPresent) {
        return;
    }
    const size_t before = triangles.size();
    triangles.erase(std::remove_if(triangles.begin(), triangles.end(),
                      [](const std::array<int, 3>& t) {
                          for (int k = 0; k < 3; ++k) {
                              if (t[k] < 0 || t[k] > kMakeHumanCanonicalLastBodyVertexInclusive) {
                                  return true;
                              }
                          }
                          return false;
                      }),
        triangles.end());
    if (triangles.empty()) {
        return;
    }
    if (before > triangles.size()) {
        std::cout << "makehuman: kept " << triangles.size() << " / " << before
                  << " triangles (body vertices 0-" << kMakeHumanCanonicalLastBodyVertexInclusive
                  << " only; helpers / cloth deformers removed).\n";
    }
}

bool loadObjIndexed(const std::string& path, std::vector<glm::vec3>& positions, std::vector<std::array<int, 3>>& triangles) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    positions.clear();
    triangles.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("v ", 0) == 0) {
            std::istringstream iss(line);
            char c = 0;
            glm::vec3 p;
            iss >> c >> p.x >> p.y >> p.z;
            positions.push_back(p);
            continue;
        }
        if (line.rfind("f ", 0) == 0) {
            std::istringstream iss(line);
            char fc = 0;
            iss >> fc;
            std::vector<int> ids;
            std::string token;
            while (iss >> token) {
                const size_t slash = token.find('/');
                const std::string vStr = (slash == std::string::npos) ? token : token.substr(0, slash);
                if (vStr.empty()) {
                    continue;
                }
                int idx = std::stoi(vStr);
                if (idx > 0) {
                    ids.push_back(idx - 1);
                } else if (idx < 0) {
                    ids.push_back(static_cast<int>(positions.size()) + idx);
                }
            }
            if (ids.size() < 3) {
                continue;
            }
            for (size_t i = 1; i + 1 < ids.size(); ++i) {
                const int i0 = ids[0];
                const int i1 = ids[i];
                const int i2 = ids[i + 1];
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= static_cast<int>(positions.size()) ||
                    i1 >= static_cast<int>(positions.size()) ||
                    i2 >= static_cast<int>(positions.size())) {
                    continue;
                }
                triangles.push_back({ i0, i1, i2 });
            }
        }
    }
    return !positions.empty() && !triangles.empty();
}

void accumulateNormals(std::vector<glm::vec3>& vertexNormals, const std::vector<glm::vec3>& positions, const std::vector<std::array<int, 3>>& triangles) {
    vertexNormals.assign(positions.size(), glm::vec3(0.0f));
    for (const auto& tri : triangles) {
        const glm::vec3& a = positions[tri[0]];
        const glm::vec3& b = positions[tri[1]];
        const glm::vec3& c = positions[tri[2]];
        const glm::vec3 fn = glm::cross(b - a, c - a);
        vertexNormals[tri[0]] += fn;
        vertexNormals[tri[1]] += fn;
        vertexNormals[tri[2]] += fn;
    }
    for (glm::vec3& n : vertexNormals) {
        const float len = glm::length(n);
        n = len > 1e-20f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

glm::vec3 skinTone(const BodyParameters& p) {
    const float t = std::clamp(p.genderBlend, 0.0f, 1.0f);
    // Slightly warmer, more saturated flesh so the body reads clearly on dark viewports.
    const glm::vec3 feminine(0.96f, 0.78f, 0.70f);
    const glm::vec3 masculine(0.88f, 0.69f, 0.58f);
    return glm::mix(feminine, masculine, t);
}

/** Per-vertex tint from height (feet slightly darker; subtle upper warmth, weaker for masculine). */
glm::vec3 bodyPaintByHeight(const glm::vec3& pos, float ymin, float ymax, const glm::vec3& base, float genderBlend) {
    const float span = std::max(ymax - ymin, 1e-6f);
    const float ny = (pos.y - ymin) / span;
    glm::vec3 c = base;
    if (ny < 0.20f) {
        const float k = std::clamp(ny / 0.20f, 0.0f, 1.0f);
        c *= glm::mix(glm::vec3(0.86f), glm::vec3(1.0f), k);
    }
    if (ny > 0.52f) {
        const float k = std::clamp((ny - 0.52f) / 0.48f, 0.0f, 1.0f);
        const float gb = std::clamp(genderBlend, 0.0f, 1.0f);
        const float rosy = glm::mix(0.5f, 0.08f, gb);
        c *= glm::mix(glm::vec3(1.0f), glm::vec3(1.04f, 0.98f, 0.97f), k * rosy);
    }
    return glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
}

void applyBodyDeform(std::vector<glm::vec3>& pos, const BodyParameters& p) {
    if (pos.empty()) {
        return;
    }
    glm::vec3 mn(FLT_MAX);
    glm::vec3 mx(-FLT_MAX);
    for (const glm::vec3& v : pos) {
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    const float ymin = mn.y;
    const float ymax = mx.y;
    const float yspan = std::max(ymax - ymin, 1e-6f);
    const float xzExtent = std::max(mx.x - mn.x, mx.z - mn.z);
    const float limbRadius = std::max(0.12f * yspan, 0.35f * xzExtent);

    const float femHip = glm::mix(0.12f, 0.0f, p.genderBlend);
    const float mascChest = glm::mix(0.0f, 0.10f, p.genderBlend);

    for (glm::vec3& v : pos) {
        const float ny = (v.y - ymin) / yspan;
        const float hipZone = std::clamp((0.38f - ny) / 0.38f, 0.0f, 1.0f);
        const float waistZone = std::exp(-std::pow((ny - 0.44f) / 0.07f, 2.0f));
        const float chestZone =
            std::clamp((ny - 0.48f) / 0.22f, 0.0f, 1.0f) * (1.0f - std::clamp((ny - 0.72f) / 0.12f, 0.0f, 1.0f));
        const float headZone = std::clamp((ny - 0.78f) / 0.22f, 0.0f, 1.0f);

        float sx = 1.0f;
        float sz = 1.0f;
        sx += 0.28f * hipZone * (p.hips - 1.0f + femHip);
        sz += 0.18f * hipZone * (p.hips - 1.0f + femHip * 0.5f);
        sx *= glm::mix(1.0f, 1.0f / std::max(p.waist, 0.35f), waistZone * 0.55f);
        sz *= glm::mix(1.0f, 1.0f / std::max(p.waist, 0.35f), waistZone * 0.55f);
        sx += 0.22f * chestZone * (p.chest - 1.0f + mascChest);
        sz += 0.12f * chestZone * (p.chest - 1.0f);

        const float headProtect = std::clamp(1.0f - headZone, 0.15f, 1.0f);
        const float w = glm::mix(1.0f, std::max(p.weight, 0.4f), headProtect);
        v.x *= sx * w;
        v.z *= sz * w;

        const float headBoost = glm::mix(1.0f, p.headScale, headZone);
        v.x *= headBoost;
        v.y = ymin + (v.y - ymin) * glm::mix(1.0f, p.headScale, headZone * 0.85f);
        v.z *= headBoost;

        const float radial = std::sqrt(v.x * v.x + v.z * v.z);
        const bool limbLike = radial > limbRadius && ny > 0.08f && ny < 0.82f;
        if (limbLike) {
            const float legBlend = std::clamp((0.52f - ny) / 0.48f, 0.0f, 1.0f);
            const float armBlend = std::clamp((ny - 0.48f) / 0.30f, 0.0f, 1.0f) * (1.0f - headZone);
            const float stretch = glm::mix(1.0f, p.legLength, legBlend) * glm::mix(1.0f, p.armLength, armBlend * 0.85f);
            const float pivotY = glm::mix(ymin + 0.55f * yspan, ymin + 0.62f * yspan, armBlend);
            v.y = pivotY + (v.y - pivotY) * stretch;
        }
    }
}

std::vector<MeshVertex> meshFromPositions(
    const std::vector<glm::vec3>& positions,
    const std::vector<std::array<int, 3>>& triangles,
    const glm::vec3& baseSkin,
    float genderBlend) {
    std::vector<glm::vec3> normals;
    accumulateNormals(normals, positions, triangles);
    glm::vec3 mn(FLT_MAX);
    glm::vec3 mx(-FLT_MAX);
    for (const glm::vec3& v : positions) {
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    const float ymin = mn.y;
    const float ymax = mx.y;
    std::vector<MeshVertex> out;
    out.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        for (int k = 0; k < 3; ++k) {
            const int idx = tri[k];
            const glm::vec3 c = bodyPaintByHeight(positions[idx], ymin, ymax, baseSkin, genderBlend);
            const glm::vec3 lit = bakeVertexShading(c, normals[idx]);
            out.push_back({ positions[idx], normals[idx], lit });
        }
    }
    return out;
}

void centerAndScaleToHeight(std::vector<glm::vec3>& positions, float heightMeters) {
    if (positions.empty()) {
        return;
    }
    glm::vec3 mn(FLT_MAX);
    glm::vec3 mx(-FLT_MAX);
    for (const glm::vec3& v : positions) {
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    const glm::vec3 center = 0.5f * (mn + mx);
    for (glm::vec3& v : positions) {
        v -= center;
    }
    mn -= center;
    mx -= center;
    const float h = mx.y - mn.y;
    const float s = (h > 1e-8f) ? (heightMeters / h) : 1.0f;
    for (glm::vec3& v : positions) {
        v *= s;
    }
}

std::vector<MeshVertex> buildParametricMannequin(const BodyParameters& p) {
    std::vector<MeshVertex> out;
    const glm::vec3 skin = skinTone(p);
    const float H = std::max(p.heightMeters, 0.5f);
    const glm::mat4 I(1.0f);

    const float headR = 0.085f * H * p.headScale;
    const float torsoRx = 0.14f * H * p.chest * glm::mix(1.05f, 0.95f, p.genderBlend);
    const float torsoRz = 0.10f * H * p.weight;
    const float hipRx = 0.15f * H * p.hips * glm::mix(1.08f, 1.0f, 1.0f - p.genderBlend);

    glm::mat4 pelvis = glm::translate(I, glm::vec3(0.0f, 0.42f * H, 0.0f)) * glm::scale(I, glm::vec3(hipRx, 0.11f * H, hipRx * 0.82f));
    appendUvSphere(out, pelvis, 1.0f, skin, 10, 14);

    glm::mat4 belly = glm::translate(I, glm::vec3(0.0f, 0.56f * H, 0.0f))
        * glm::scale(I, glm::vec3(torsoRx / std::max(p.waist, 0.45f), 0.14f * H, torsoRz / std::max(p.waist, 0.45f)));
    appendUvSphere(out, belly, 1.0f, skin, 10, 14);

    glm::mat4 chest = glm::translate(I, glm::vec3(0.0f, 0.72f * H, 0.0f)) * glm::scale(I, glm::vec3(torsoRx, 0.12f * H, torsoRz));
    appendUvSphere(out, chest, 1.0f, skin, 10, 14);

    glm::mat4 head = glm::translate(I, glm::vec3(0.0f, 0.92f * H, 0.0f));
    const glm::vec3 headRosy = glm::clamp(skin * glm::vec3(1.04f, 0.98f, 0.98f), glm::vec3(0.0f), glm::vec3(1.0f));
    const float gb = std::clamp(p.genderBlend, 0.0f, 1.0f);
    const glm::vec3 headSkin = glm::mix(headRosy, skin, gb);
    appendUvSphere(out, head, headR, headSkin, 12, 16);

    const float legScale = p.legLength;
    const float armScale = p.armLength;
    const float thighR = 0.055f * H * std::sqrt(p.weight);
    const float shinR = 0.045f * H * std::sqrt(p.weight);
    const float upperArmR = 0.038f * H * std::sqrt(p.weight);
    const float foreArmR = 0.032f * H * std::sqrt(p.weight);

    for (float side : { -1.0f, 1.0f }) {
        const float xLeg = side * 0.09f * H * p.chest;
        appendYCylinder(out, glm::vec3(xLeg, 0.33f * H * legScale, 0.0f), thighR, 0.22f * H * legScale, skin * 0.95f, 12, 4);
        appendYCylinder(out, glm::vec3(xLeg, 0.13f * H * legScale, 0.0f), shinR, 0.24f * H * legScale, skin * 0.92f, 10, 4);

        const float xArm = side * (0.12f * H * p.chest + 0.02f * H);
        const float yShoulder = 0.78f * H;
        appendYCylinder(out, glm::vec3(xArm, yShoulder - 0.08f * H * armScale, 0.0f), upperArmR, 0.16f * H * armScale, skin * 0.93f, 10, 3);
        appendYCylinder(out, glm::vec3(xArm, yShoulder - 0.24f * H * armScale, 0.0f), foreArmR, 0.15f * H * armScale,
            glm::clamp(skin * glm::vec3(0.94f, 0.90f, 0.88f), glm::vec3(0.0f), glm::vec3(1.0f)), 8, 3);
    }

    return out;
}

} // namespace

std::vector<std::string> defaultMakeHumanRoots() {
    return {
        std::string(kGitMakeHumanRoot),
        std::string(kLegacyInstallerRoot),
    };
}

std::string resolveBaseObjPath() {
    const std::string candidate = "models/makehuman.obj";
    std::ifstream probe(candidate);
    if (probe) {
        return candidate;
    }
    return {};
}

std::vector<MeshVertex> buildHumanMesh(const BodyParameters& params) {
    const std::string obj = resolveBaseObjPath();
    std::vector<glm::vec3> positions;
    std::vector<std::array<int, 3>> triangles;
    if (!obj.empty() && loadObjIndexed(obj, positions, triangles)) {
        std::vector<std::array<int, 3>> trisFull = triangles;
        filterToCanonicalBodyTriangles(triangles, positions.size());
        if (triangles.empty()) {
            triangles = std::move(trisFull);
            std::cout << "makehuman: body-only vertex filter removed all faces; using full OBJ.\n";
        }
        applyBodyDeform(positions, params);
        centerAndScaleToHeight(positions, params.heightMeters);
        return meshFromPositions(positions, triangles, skinTone(params), params.genderBlend);
    }
    std::cout << "makehuman: base.obj not found or empty; using parametric mannequin.\n";
    return buildParametricMannequin(params);
}

} // namespace makehuman
