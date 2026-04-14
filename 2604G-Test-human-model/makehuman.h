#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

/// Helpers for displaying a MakeHuman-style body in the Test viewer.
/// Asset roots match local clones under `resources/makehuman/…`.
namespace makehuman {

/// Canonical locations (git checkout + legacy installer folder).
std::vector<std::string> defaultMakeHumanRoots();

/// First existing file among known candidates (typically `…/makehuman/data/3dobjs/base.obj`).
std::string resolveBaseObjPath();

/// Tunable body metrics (not a full MakeHuman target system; drives coarse mesh edits).
/// Default mesh is an adult **male**, nude-style body (MakeHuman `base.obj` is the neutral
/// unclothed reference; this app does not load clothing assets).
struct BodyParameters {
    /// Target standing height in metres after normalization.
    float heightMeters = 1.78f;
    /// 0 = more feminine proportions, 1 = more masculine (shoulder/hip bias).
    float genderBlend = 1.0f;
    /// Multipliers around 1: breadth across chest / upper torso.
    float chest = 1.0f;
    /// <1 narrows waist, >1 widens.
    float waist = 1.0f;
    /// Pelvis / upper leg lateral width.
    float hips = 1.0f;
    /// Overall thickness in the horizontal plane (excluding most of the head).
    float weight = 1.0f;
    /// Stretches vertices that read as limbs along vertical bones (heuristic).
    float armLength = 1.0f;
    float legLength = 1.0f;
    float headScale = 1.0f;
};

/// Vertex layout matches `Vertex` in VulkanWindow.h (pos, normal, color).
/// For the human mesh, `color` includes **baked diffuse lighting** so the body stays readable even
/// when only precompiled fragment SPIR-V is used (see `bakeVertexShading` in makehuman.cpp).
struct MeshVertex {
    glm::vec3 pos{};
    glm::vec3 normal{};
    glm::vec3 color{};
};

/// Loads `base.obj` when present, applies `BodyParameters`, recenters, scales to `heightMeters`.
/// Falls back to a simple parametric mannequin if the file cannot be read.
std::vector<MeshVertex> buildHumanMesh(const BodyParameters& params);

} // namespace makehuman
