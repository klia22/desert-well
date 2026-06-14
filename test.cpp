// Verification program. Plug in worldSeed, chunkX, and chunkZ for the origin chunk, and if you see 4 
// "Found well at world coordinates" messages, the coordinate is correct.
#include <bits/stdc++.h>
namespace DW {

constexpr uint32_t imul32(uint32_t a, uint32_t b) noexcept {
    int64_t x = int64_t(int32_t(a)) * int64_t(int32_t(b));
    return uint32_t(x);
}

struct MTRandom {
    uint32_t mt[624];
    int index = 624;

    explicit MTRandom(uint32_t seed) noexcept {
        seed32(seed);
    }

    void seed32(uint32_t seed) noexcept {
        mt[0] = seed;
        for (int i = 1; i < 624; ++i) {
            uint32_t prev = mt[i - 1];
            mt[i] = uint32_t(imul32(1812433253u, prev ^ (prev >> 30)) + uint32_t(i));
        }
        index = 624;
    }

    void twist() noexcept {
        for (int i = 0; i < 624; ++i) {
            uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % 624] & 0x7fffffffu);
            uint32_t m = mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1u) {
                m ^= 0x9908b0dfu;
            }
            mt[i] = m;
        }
        index = 0;
    }

    uint32_t random_int() noexcept {
        if (index >= 624) {
            twist();
        }
        uint32_t y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 2636928640u;
        y ^= (y << 15) & 4022730752u;
        y ^= (y >> 18);
        return y;
    }
};

struct RNG {
    MTRandom rng;
    explicit RNG(uint32_t seed) noexcept : rng(seed) {}

    uint32_t next31() noexcept {
        return rng.random_int() >> 1;
    }

    template <uint32_t Bound>
    uint32_t next() noexcept {
        return rng.random_int() % Bound;
    }

    uint32_t next(uint32_t bound) noexcept {
        return rng.random_int() % bound;
    }
};

constexpr uint32_t strHash(const char *text) noexcept {
    int32_t value = -2078137563;
    while (*text) {
        value = int32_t(imul32(uint32_t(value), 435u) ^ int32_t(static_cast<unsigned char>(*text++)));
    }
    return uint32_t(value);
}

static constexpr uint32_t FEATURE_KEY = strHash("minecraft:desert_after_surface_desert_well_feature");

struct FeatureSeed {
    uint32_t seedLow32;
    uint32_t xMul;
    uint32_t zMul;
    uint32_t fKey;
};

inline FeatureSeed DWMakeFeatureSeed(int64_t worldSeed) noexcept {
    uint32_t seedLow32 = uint32_t(int32_t(worldSeed));
    RNG rng(seedLow32);
    return FeatureSeed{
        seedLow32,
        rng.next31() | 1u,
        rng.next31() | 1u,
        DW::FEATURE_KEY
    };
}

inline uint32_t fHash(const FeatureSeed &seed, int32_t chunkX, int32_t chunkZ) noexcept {
    int32_t combined = int32_t(imul32(uint32_t(chunkX), seed.xMul))
                          + int32_t(imul32(uint32_t(chunkZ), seed.zMul));
    uint32_t base = uint32_t(uint32_t(seed.seedLow32) ^ uint32_t(combined));
    return uint32_t(base ^ (seed.fKey + (base << 6) + (base >> 2) - 1640531527u));
}

struct WellResult {
    bool hasWell;
    int worldX;
    int worldZ;
};

inline WellResult findwell(int64_t worldSeed, int chunkX, int chunkZ) noexcept {
    FeatureSeed seed = DWMakeFeatureSeed(worldSeed);
    uint32_t regionSeed = fHash(seed, chunkX, chunkZ);
    RNG rng(regionSeed);
    if (rng.next<500>() != 0u) {
        return {false, 0, 0};
    }
    int offsetZ = int(rng.next<16>());
    int offsetX = int(rng.next<16>());
    return {true, chunkX * 16 + offsetX, chunkZ * 16 + offsetZ};
}

inline bool isWell(int64_t worldSeed, int chunkX, int chunkZ) noexcept {
    return findwell(worldSeed, chunkX, chunkZ).hasWell;
}

}
using namespace std;
using namespace DW;
int main() {
    uint32_t worldSeed = 536879937;
    uint32_t chunkX = 26737318;
    uint32_t chunkZ = -3496929;
    for(int ax = chunkX - 1; ax <= chunkX + 1; ++ax) {
        for(int az = chunkZ - 1; az <= chunkZ + 1; ++az) {
            auto result = findwell(worldSeed, ax, az);
            if(result.hasWell) {
                cout << "Found well at world coordinates (" << result.worldX << ", " << result.worldZ << ") "<< "Chunk (" << ax << ", " << az << ")\n";
            } else {
                cout << "No well at chunk (" << ax << ", " << az << ")\n";
            }
        }
    }
}