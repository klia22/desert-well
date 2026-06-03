#ifndef WELL_H
#define WELL_H

#include <cstdint>

namespace DW {

constexpr std::uint32_t imul32(std::uint32_t a, std::uint32_t b) noexcept {
    std::int64_t x = std::int64_t(std::int32_t(a)) * std::int64_t(std::int32_t(b));
    return std::uint32_t(x);
}

struct MTRandom {
    std::uint32_t mt[624];
    int index = 624;

    explicit MTRandom(std::uint32_t seed) noexcept {
        seed32(seed);
    }

    void seed32(std::uint32_t seed) noexcept {
        mt[0] = seed;
        for (int i = 1; i < 624; ++i) {
            std::uint32_t prev = mt[i - 1];
            mt[i] = std::uint32_t(imul32(1812433253u, prev ^ (prev >> 30)) + std::uint32_t(i));
        }
        index = 624;
    }

    void twist() noexcept {
        for (int i = 0; i < 624; ++i) {
            std::uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % 624] & 0x7fffffffu);
            std::uint32_t m = mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1u) {
                m ^= 0x9908b0dfu;
            }
            mt[i] = m;
        }
        index = 0;
    }

    std::uint32_t random_int() noexcept {
        if (index >= 624) {
            twist();
        }
        std::uint32_t y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 2636928640u;
        y ^= (y << 15) & 4022730752u;
        y ^= (y >> 18);
        return y;
    }
};

struct RandWrapper {
    MTRandom rng;
    explicit RandWrapper(std::uint32_t seed) noexcept : rng(seed) {}

    std::uint32_t nextInt31() noexcept {
        return rng.random_int() >> 1;
    }

    template <std::uint32_t Bound>
    std::uint32_t nextInt() noexcept {
        return rng.random_int() % Bound;
    }

    std::uint32_t nextInt(std::uint32_t bound) noexcept {
        return rng.random_int() % bound;
    }
};

constexpr std::uint32_t stringHashConst(const char *text) noexcept {
    std::int32_t value = -2078137563;
    while (*text) {
        value = std::int32_t(imul32(std::uint32_t(value), 435u) ^ std::int32_t(static_cast<unsigned char>(*text++)));
    }
    return std::uint32_t(value);
}

static constexpr std::uint32_t FEATURE_KEY = stringHashConst("minecraft:desert_after_surface_desert_well_feature");

struct FeatureSeed {
    std::uint32_t seedLow32;
    std::uint32_t xMul;
    std::uint32_t zMul;
    std::uint32_t featureKey;
};

inline FeatureSeed makeFeatureSeed(std::int64_t worldSeed) noexcept {
    std::uint32_t seedLow32 = std::uint32_t(std::int32_t(worldSeed));
    RandWrapper rng(seedLow32);
    return FeatureSeed{
        seedLow32,
        rng.nextInt31() | 1u,
        rng.nextInt31() | 1u,
        FEATURE_KEY
    };
}

inline std::uint32_t featureSeedHash(const FeatureSeed &seed, std::int32_t chunkX, std::int32_t chunkZ) noexcept {
    std::int32_t combined = std::int32_t(imul32(std::uint32_t(chunkX), seed.xMul))
                          + std::int32_t(imul32(std::uint32_t(chunkZ), seed.zMul));
    std::uint32_t base = std::uint32_t(std::uint32_t(seed.seedLow32) ^ std::uint32_t(combined));
    return std::uint32_t(base ^ (seed.featureKey + (base << 6) + (base >> 2) - 1640531527u));
}

struct WellResult {
    bool hasWell;
    int worldX;
    int worldZ;
};

inline WellResult findwell(std::int64_t worldSeed, int chunkX, int chunkZ) noexcept {
    FeatureSeed seed = makeFeatureSeed(worldSeed);
    std::uint32_t regionSeed = featureSeedHash(seed, chunkX, chunkZ);
    RandWrapper rng(regionSeed);
    if (rng.nextInt<500>() != 0u) {
        return {false, 0, 0};
    }
    int offsetZ = int(rng.nextInt<16>());
    int offsetX = int(rng.nextInt<16>());
    return {true, chunkX * 16 + offsetX, chunkZ * 16 + offsetZ};
}

inline bool isWell(std::int64_t worldSeed, int chunkX, int chunkZ) noexcept {
    return findwell(worldSeed, chunkX, chunkZ).hasWell;
}

} // namespace DW

#endif // WELL_H
