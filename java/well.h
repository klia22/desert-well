#include <bits/stdc++.h>
#include <optional>
#include <cstdint>
using namespace std;
#ifndef GEODE_H
#define GEODE_H

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

struct JavaRandom {
    uint64_t lo;
    uint64_t hi;

    void setSeed(uint64_t value);

    uint64_t nextLong();

    int nextIntJ(uint32_t n);

    float nextFloat();

    void skipN(int count);

    uint64_t nextLongJ();
};

static inline uint64_t getPopulationSeed(uint64_t ws, int x, int z) {
    JavaRandom xr;
    xr.setSeed(ws);
    uint64_t a = xr.nextLongJ();
    uint64_t b = xr.nextLongJ();
    a |= 1ULL;
    b |= 1ULL;
    return (static_cast<uint64_t>(x) * a + static_cast<uint64_t>(z) * b) ^ ws;
}

struct GeodeResult {
    int worldX = 0;
    int worldY = 0;
    int worldZ = 0;
    int size = 0;
    bool cracked = false;
};

static inline optional<GeodeResult> findGeode(uint64_t worldSeed, int chunkX, int chunkZ, uint64_t populationseed = 0) {
    const uint64_t salt = 20002ULL;
    const float rarity = 1.0f / 24.0f;
    int blockX = chunkX * 16;
    int blockZ = chunkZ * 16;
    int regionX = blockX & ~15;
    int regionZ = blockZ & ~15;

    JavaRandom xr;
    xr.setSeed(populationseed == 0 ? getPopulationSeed(worldSeed, regionX, regionZ) + salt : populationseed + salt);
    if (xr.nextFloat() >= rarity) {
        return nullopt;
    }

    int rx = xr.nextIntJ(16);
    int rz = xr.nextIntJ(16);
    rx -= blockX & 15;
    rz -= blockZ & 15;
    int ry = xr.nextIntJ(89) - 58;
    int size = xr.nextIntJ(2) + 3;
    xr.skipN(2);
    bool cracked = xr.nextFloat() < 0.95f;

    rx += 4;
    ry += 4;
    rz += 4;

    GeodeResult result;
    result.worldX = regionX + rx;
    result.worldY = ry;
    result.worldZ = regionZ + rz;
    result.size = size;
    result.cracked = cracked;
    return result;
}



bool isGeode(uint64_t ws, int cx, int cz, uint64_t popseed);
#endif