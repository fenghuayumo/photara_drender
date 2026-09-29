// Standalone UVAtlas timing harness used to profile Microsoft UVAtlas.
// Generates manifold test meshes and reports UVAtlasCreate phase timings.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <array>
#include <set>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <functional>

#include <DirectXMath.h>
#include <UVAtlas.h>

using namespace DirectX;

namespace
{

struct Mesh
{
    std::vector<XMFLOAT3> positions;
    std::vector<uint32_t> indices;
    std::string name;
};

Mesh make_sphere(int rings, int segments)
{
    Mesh mesh;
    mesh.name = "sphere";
    mesh.positions.reserve(size_t(rings - 1) * segments + 2);
    mesh.positions.emplace_back(0.f, 1.f, 0.f);
    for (int i = 1; i < rings; ++i)
    {
        const float phi = DirectX::XM_PI * float(i) / float(rings);
        for (int j = 0; j < segments; ++j)
        {
            const float theta = 2.f * DirectX::XM_PI * float(j) / float(segments);
            mesh.positions.emplace_back(
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta));
        }
    }
    mesh.positions.emplace_back(0.f, -1.f, 0.f);

    const uint32_t north = 0;
    const uint32_t south = uint32_t(mesh.positions.size() - 1);
    const auto vidx = [segments](int ring, int slot) -> uint32_t
    {
        // ring 1 .. rings-1, slot 0 .. segments-1
        return uint32_t((ring - 1) * segments + 1 + slot);
    };
    for (int j = 0; j < segments; ++j)
    {
        const uint32_t a = vidx(1, j);
        const uint32_t b = vidx(1, (j + 1) % segments);
        mesh.indices.insert(mesh.indices.end(), { north, b, a });
        mesh.indices.insert(mesh.indices.end(),
            { south, vidx(rings - 1, j), vidx(rings - 1, (j + 1) % segments) });
    }
    for (int i = 1; i < rings - 1; ++i)
    {
        for (int j = 0; j < segments; ++j)
        {
            const uint32_t a = vidx(i, j);
            const uint32_t b = vidx(i, (j + 1) % segments);
            const uint32_t c = vidx(i + 1, j);
            const uint32_t d = vidx(i + 1, (j + 1) % segments);
            mesh.indices.insert(mesh.indices.end(), { a, c, b });
            mesh.indices.insert(mesh.indices.end(), { b, c, d });
        }
    }
    return mesh;
}

Mesh make_torus(int rings, int segments)
{
    Mesh mesh;
    mesh.name = "torus";
    mesh.positions.reserve(size_t(rings) * segments);
    for (int i = 0; i < rings; ++i)
    {
        const float u = 2.f * DirectX::XM_PI * float(i) / float(rings);
        for (int j = 0; j < segments; ++j)
        {
            const float v = 2.f * DirectX::XM_PI * float(j) / float(segments);
            const float r = 1.f + 0.35f * std::cos(v);
            mesh.positions.emplace_back(r * std::cos(u), 0.35f * std::sin(v), r * std::sin(u));
        }
    }
    for (int i = 0; i < rings; ++i)
    {
        for (int j = 0; j < segments; ++j)
        {
            const uint32_t a = uint32_t(i) * segments + uint32_t(j);
            const uint32_t b = uint32_t(i) * segments + uint32_t((j + 1) % segments);
            const uint32_t c = uint32_t((i + 1) % rings) * segments + uint32_t(j);
            const uint32_t d = uint32_t((i + 1) % rings) * segments + uint32_t((j + 1) % segments);
            mesh.indices.insert(mesh.indices.end(), { a, c, b });
            mesh.indices.insert(mesh.indices.end(), { b, c, d });
        }
    }
    return mesh;
}

float hash_noise(float x, float y)
{
    float value = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return value - std::floor(value);
}

float smooth_noise(float x, float y)
{
    float sum = 0.f;
    float amp = 1.f;
    float freq = 1.f;
    for (int o = 0; o < 4; ++o)
    {
        sum += amp * hash_noise(x * freq, y * freq);
        amp *= 0.5f;
        freq *= 2.1f;
    }
    return sum;
}

Mesh make_terrain(int width, int height)
{
    Mesh mesh;
    mesh.name = "terrain";
    mesh.positions.reserve(size_t(width) * height);
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            const float x = float(i) / float(width - 1);
            const float y = float(j) / float(height - 1);
            const float r = std::sqrt((x - .5f) * (x - .5f) + (y - .5f) * (y - .5f));
            const float bump = r < 0.5f ? std::cos(r * DirectX::XM_PI) * 0.25f : 0.f;
            mesh.positions.emplace_back(x * 2.f - 1.f,
                smooth_noise(x * 8.f, y * 8.f) * 0.45f + bump,
                y * 2.f - 1.f);
        }
    }
    for (int j = 0; j + 1 < height; ++j)
    {
        for (int i = 0; i + 1 < width; ++i)
        {
            const uint32_t a = uint32_t(j) * width + uint32_t(i);
            const uint32_t b = a + 1;
            const uint32_t c = a + uint32_t(width);
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), { a, c, b });
            mesh.indices.insert(mesh.indices.end(), { b, c, d });
        }
    }
    return mesh;
}

Mesh make_shells(int shells, int rings, int segments)
{
    Mesh mesh;
    mesh.name = "shells";
    for (int s = 0; s < shells; ++s)
    {
        Mesh part = make_sphere(rings, segments);
        const uint32_t offset = uint32_t(mesh.positions.size());
        const float scale = 0.5f + 0.4f * hash_noise(float(s), 1.7f);
        const float dx = -1.5f + 3.f * (float(s) / std::max(1, shells - 1));
        for (auto& p : part.positions)
        {
            mesh.positions.emplace_back(p.x * scale + dx, p.y * scale, p.z * scale);
        }
        for (auto idx : part.indices)
        {
            mesh.indices.push_back(idx + offset);
        }
    }
    return mesh;
}

Mesh load_obj(const std::string& path)
{
    Mesh mesh;
    mesh.name = "obj";
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(2);
    }
    std::string line;
    size_t polygons = 0;
    while (std::getline(in, line))
    {
        if (line.compare(0, 2, "v ") == 0)
        {
            std::istringstream ss(line.substr(2));
            XMFLOAT3 p;
            ss >> p.x >> p.y >> p.z;
            mesh.positions.push_back(p);
        }
        else if (line.compare(0, 2, "f ") == 0)
        {
            std::istringstream ss(line.substr(2));
            std::vector<uint32_t> face;
            std::string token;
            while (ss >> token)
            {
                const size_t slash = token.find('/');
                const std::string indexText = slash == std::string::npos ? token : token.substr(0, slash);
                const long long raw = std::stoll(indexText);
                const long long index = raw > 0 ? raw - 1 : static_cast<long long>(mesh.positions.size()) + raw;
                face.push_back(static_cast<uint32_t>(index));
            }
            if (face.size() < 3)
            {
                continue;
            }
            if (face.size() != 3) ++polygons;
            for (size_t i = 2; i < face.size(); ++i)
            {
                mesh.indices.insert(mesh.indices.end(), { face[0], face[i - 1], face[i] });
            }
        }
    }
    std::fprintf(stderr, "loaded OBJ %s: %zu vertices, %zu triangles (%zu non-triangle faces)\n",
        path.c_str(), mesh.positions.size(), mesh.indices.size() / 3, polygons);
    return mesh;
}

Mesh load_ply(const std::string& path)
{
    Mesh mesh;
    mesh.name = "ply";
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(2);
    }

    auto typeSize = [](const std::string& type) -> size_t
    {
        if (type == "char" || type == "uchar") return 1;
        if (type == "short" || type == "ushort") return 2;
        if (type == "int" || type == "uint" || type == "float") return 4;
        if (type == "double") return 8;
        return 0;
    };

    std::string line;
    size_t vertexCount = 0;
    size_t faceCount = 0;
    struct VertexProperty
    {
        std::string name;
        size_t offset;
        size_t size;
    };
    std::vector<VertexProperty> vertexProperties;
    size_t vertexStride = 0;
    std::string faceListIndexType = "int";
    std::string faceListCountType = "uchar";
    std::string element;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "format")
        {
            std::string encoding;
            ss >> encoding;
            if (encoding.compare(0, 6, "binary") != 0)
            {
                std::fprintf(stderr, "only binary PLY files are supported: %s\n", encoding.c_str());
                std::exit(2);
            }
        }
        else if (tag == "element")
        {
            ss >> element;
            if (element == "vertex") ss >> vertexCount;
            else if (element == "face") ss >> faceCount;
        }
        else if (tag == "property")
        {
            std::string type;
            std::string name;
            ss >> type;
            if (type == "list")
            {
                ss >> faceListCountType >> faceListIndexType >> name;
            }
            else
            {
                ss >> name;
                if (element == "vertex")
                {
                    const size_t size = typeSize(type);
                    vertexProperties.push_back({ name, vertexStride, size });
                    vertexStride += size;
                }
            }
        }
        else if (tag == "end_header")
        {
            break;
        }
    }

    auto readLE = [&in](void* dst, size_t size)
    {
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
    };

    size_t xOffset = SIZE_MAX;
    size_t yOffset = SIZE_MAX;
    size_t zOffset = SIZE_MAX;
    size_t xSize = 4, ySize = 4, zSize = 4;
    for (const auto& prop : vertexProperties)
    {
        if (prop.name == "x") { xOffset = prop.offset; xSize = prop.size; }
        else if (prop.name == "y") { yOffset = prop.offset; ySize = prop.size; }
        else if (prop.name == "z") { zOffset = prop.offset; zSize = prop.size; }
    }
    if (xOffset == SIZE_MAX || yOffset == SIZE_MAX || zOffset == SIZE_MAX || vertexStride == 0)
    {
        std::fprintf(stderr, "PLY has no x/y/z vertex properties\n");
        std::exit(2);
    }

    mesh.positions.resize(vertexCount);
    std::vector<char> vertex(vertexStride);
    for (size_t i = 0; i < vertexCount; ++i)
    {
        readLE(vertex.data(), vertexStride);
        const auto readCoord = [&](size_t offset, size_t size) -> float
        {
            return size == 8
                ? static_cast<float>(*reinterpret_cast<const double*>(vertex.data() + offset))
                : *reinterpret_cast<const float*>(vertex.data() + offset);
        };
        mesh.positions[i] = XMFLOAT3(
            readCoord(xOffset, xSize),
            readCoord(yOffset, ySize),
            readCoord(zOffset, zSize));
    }

    const size_t countSize = typeSize(faceListCountType);
    const size_t indexSize = typeSize(faceListIndexType);
    for (size_t i = 0; i < faceCount; ++i)
    {
        uint64_t count = 0;
        if (countSize == 1) { uint8_t v; readLE(&v, 1); count = v; }
        else if (countSize == 2) { uint16_t v; readLE(&v, 2); count = v; }
        else { uint32_t v; readLE(&v, 4); count = v; }
        std::vector<uint32_t> face(static_cast<size_t>(count));
        for (size_t j = 0; j < face.size(); ++j)
        {
            if (indexSize == 2) { uint16_t v; readLE(&v, 2); face[j] = v; }
            else if (indexSize == 8) { uint64_t v; readLE(&v, 8); face[j] = static_cast<uint32_t>(v); }
            else { uint32_t v; readLE(&v, 4); face[j] = v; }
        }
        for (size_t j = 2; j < face.size(); ++j)
        {
            mesh.indices.insert(mesh.indices.end(), { face[0], face[j - 1], face[j] });
        }
    }
    std::fprintf(stderr, "loaded PLY %s: %zu vertices, %zu triangles\n",
        path.c_str(), mesh.positions.size(), mesh.indices.size() / 3);
    return mesh;
}

std::vector<uint32_t> build_adjacency(const Mesh& mesh)
{
    struct EdgeKey
    {
        uint32_t a, b;
        bool operator<(const EdgeKey& o) const
        {
            return std::tie(a, b) < std::tie(o.a, o.b);
        }
    };
    std::map<EdgeKey, std::vector<uint32_t>> edges;
    const size_t faces = mesh.indices.size() / 3;
    for (size_t f = 0; f < faces; ++f)
    {
        for (int e = 0; e < 3; ++e)
        {
            uint32_t a = mesh.indices[f * 3 + e];
            uint32_t b = mesh.indices[f * 3 + (e + 1) % 3];
            if (a == b)
            {
                std::fprintf(stderr, "degenerate edge in face %zu\n", f);
            }
            if (a > b) std::swap(a, b);
            edges[{ a, b }].push_back(uint32_t(f));
        }
    }
    std::vector<uint32_t> adjacency(faces * 3, uint32_t(-1));
    size_t boundary = 0;
    for (size_t f = 0; f < faces; ++f)
    {
        for (int e = 0; e < 3; ++e)
        {
            uint32_t a = mesh.indices[f * 3 + e];
            uint32_t b = mesh.indices[f * 3 + (e + 1) % 3];
            if (a > b) std::swap(a, b);
            auto it = edges.find({ a, b });
            uint32_t other = uint32_t(-1);
            if (it != edges.end())
            {
                for (uint32_t cand : it->second)
                {
                    if (cand != f) { other = cand; break; }
                }
            }
            adjacency[f * 3 + e] = other;
            if (other == uint32_t(-1)) ++boundary;
        }
    }
    std::fprintf(stderr, "mesh=%s verts=%zu faces=%zu boundary_edges=%zu\n",
        mesh.name.c_str(), mesh.positions.size(), faces, boundary);
    return adjacency;
}

// Diagnostic helper: drop exact-duplicate faces and enough extra faces so
// every edge is shared by at most two faces. This mimics a manifold repair
// pass so UVAtlas can be timed on manifold input.
void make_manifold(Mesh& mesh)
{
    struct EdgeState
    {
        uint32_t count = 0;
        uint32_t keepFaces[2] = { uint32_t(-1), uint32_t(-1) };
    };
    std::map<std::pair<uint32_t, uint32_t>, EdgeState> edges;
    std::set<std::array<uint32_t, 3>> faceSet;

    std::vector<uint32_t> kept;
    kept.reserve(mesh.indices.size());
    const size_t faces = mesh.indices.size() / 3;
    size_t droppedDuplicate = 0;
    size_t droppedNonmanifold = 0;
    for (size_t f = 0; f < faces; ++f)
    {
        std::array<uint32_t, 3> tri = {
            mesh.indices[f * 3],
            mesh.indices[f * 3 + 1],
            mesh.indices[f * 3 + 2]
        };
        std::array<uint32_t, 3> key = tri;
        std::sort(key.begin(), key.end());
        if (!faceSet.insert(key).second)
        {
            ++droppedDuplicate;
            continue;
        }

        bool drop = false;
        std::pair<uint32_t, uint32_t> edgeKeys[3];
        for (int e = 0; e < 3; ++e)
        {
            uint32_t a = tri[e];
            uint32_t b = tri[(e + 1) % 3];
            if (a > b) std::swap(a, b);
            edgeKeys[e] = { a, b };
            if (edges[edgeKeys[e]].count >= 2)
            {
                drop = true;
            }
        }
        if (drop)
        {
            ++droppedNonmanifold;
            continue;
        }
        for (int e = 0; e < 3; ++e)
        {
            EdgeState& state = edges[edgeKeys[e]];
            state.keepFaces[state.count++] = uint32_t(f);
        }
        kept.insert(kept.end(), tri.begin(), tri.end());
    }
    std::fprintf(stderr, "manifold fix: kept %zu faces (dropped %zu duplicate, %zu non-manifold)\n",
        kept.size() / 3, droppedDuplicate, droppedNonmanifold);
    mesh.indices = std::move(kept);
}

} // namespace

int main(int argc, char** argv)
{
    std::string kind = "sphere";
    std::string filePath;
    int target_faces = 200000;
    int atlas_size = 4096;
    float gutter = 1.0f;
    float max_stretch = 1.f / 6.f;
    uint32_t options = 0;
    int omp_threads = 0;
    bool manifoldFix = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--mesh") kind = next("--mesh");
        else if (arg == "--file") filePath = next("--file");
        else if (arg == "--faces") target_faces = std::atoi(next("--faces").c_str());
        else if (arg == "--atlas") atlas_size = std::atoi(next("--atlas").c_str());
        else if (arg == "--gutter") gutter = float(std::atof(next("--gutter").c_str()));
        else if (arg == "--max-stretch") max_stretch = float(std::atof(next("--max-stretch").c_str()));
        else if (arg == "--fast") options = UVATLAS_GEODESIC_FAST;
        else if (arg == "--quality") options = UVATLAS_GEODESIC_QUALITY;
        else if (arg == "--omp-threads") omp_threads = std::atoi(next("--omp-threads").c_str());
        else if (arg == "--manifold-fix") manifoldFix = true;
        else
        {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (omp_threads > 0)
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", omp_threads);
        _putenv_s("OMP_NUM_THREADS", buf);
    }

    Mesh mesh;
    if (!filePath.empty())
    {
        const bool isObj = filePath.size() >= 4 && filePath.compare(filePath.size() - 4, 4, ".obj") == 0;
        mesh = isObj ? load_obj(filePath) : load_ply(filePath);
        kind = "file:" + filePath;
    }
    else if (kind == "sphere")
    {
        const int seg = int(std::sqrt(double(target_faces) / 2.0));
        mesh = make_sphere(std::max(4, seg), std::max(4, seg));
    }
    else if (kind == "torus")
    {
        const int seg = int(std::sqrt(double(target_faces) / 2.0));
        mesh = make_torus(std::max(4, seg), std::max(4, seg));
    }
    else if (kind == "terrain")
    {
        const int side = int(std::sqrt(double(target_faces) / 2.0)) + 1;
        mesh = make_terrain(side, side);
    }
    else if (kind == "shells")
    {
        const int shells = 32;
        const int seg = int(std::sqrt(double(target_faces) / double(2 * shells)));
        mesh = make_shells(shells, std::max(4, seg), std::max(4, seg));
    }
    else
    {
        std::fprintf(stderr, "unknown mesh kind %s\n", kind.c_str());
        return 2;
    }

    if (manifoldFix)
    {
        make_manifold(mesh);
    }

    const std::vector<uint32_t> adjacency = build_adjacency(mesh);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    float last_reported = -1.f;
    auto callback = [&](float percentDone) -> HRESULT
    {
        const float step = 5.f;
        const float bucket = std::floor(percentDone * 100.f / step) * step;
        if (bucket > last_reported)
        {
            last_reported = bucket;
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            const double sec = double(now.QuadPart - t0.QuadPart) / double(freq.QuadPart);
            std::fprintf(stderr, "progress %5.1f%% at %8.3fs\n", percentDone * 100.f, sec);
        }
        return S_OK;
    };

    std::vector<UVAtlasVertex> outVerts;
    std::vector<uint8_t> outIndices;
    std::vector<uint32_t> facePartition;
    std::vector<uint32_t> vertexRemap;
    float stretchOut = 0.f;
    size_t chartsOut = 0;

    const HRESULT hr = UVAtlasCreate(
        mesh.positions.data(),
        mesh.positions.size(),
        mesh.indices.data(),
        DXGI_FORMAT_R32_UINT,
        mesh.indices.size() / 3,
        0,
        max_stretch,
        size_t(atlas_size),
        size_t(atlas_size),
        gutter,
        adjacency.data(),
        nullptr,
        nullptr,
        callback,
        0.0001f,
        static_cast<DirectX::UVATLAS>(options),
        outVerts,
        outIndices,
        &facePartition,
        &vertexRemap,
        &stretchOut,
        &chartsOut);

    QueryPerformanceCounter(&t1);
    const double total = double(t1.QuadPart - t0.QuadPart) / double(freq.QuadPart);
    if (FAILED(hr))
    {
        std::fprintf(stderr, "UVAtlasCreate failed: 0x%08lX\n", static_cast<unsigned long>(hr));
        return 1;
    }
    std::fprintf(stderr,
        "RESULT kind=%s faces=%zu atlas=%d gutter=%.2f stretch_in=%.4f options=%u "
        "total_sec=%.3f charts=%zu out_verts=%zu out_faces=%zu max_stretch=%.4f\n",
        kind.c_str(),
        mesh.indices.size() / 3,
        atlas_size,
        gutter,
        max_stretch,
        options,
        total,
        chartsOut,
        outVerts.size(),
        outIndices.size() / 12,
        stretchOut);
    return 0;
}
