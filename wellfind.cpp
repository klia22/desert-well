//Main search:
//Compile with -Ofast -march=native -mtune=native -flto -DNDEBUG -pthread for best results
//May take days to complete
#include <bits/stdc++.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <numeric>
#include <immintrin.h>
#include <memory>

using namespace std;
using namespace chrono;

static atomic<bool> stopRequested{false};
static atomic<uint64_t> foundCount{0};
static atomic<uint64_t> candidateCount{0};
static mutex outputMutex;

//Bedrock RNG from well.h
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

enum Corner : int {
    NW = 0,
    NE = 1,
    SW = 2,
    SE = 3,
    CORNER_COUNT = 4
};

static constexpr uint64_t TOTAL_UINT32 = 0x1'0000'0000ULL;
static constexpr uint32_t REPORT_INTERVAL_SECONDS = 20;
static constexpr uint64_t FLUSH_INTERVAL = 20;

// Exact low22 filtering
static constexpr int LOWER_BITS = 22;
static constexpr uint32_t LOWER_SIZE = 1u << LOWER_BITS;
static constexpr uint32_t LOWER_MASK = LOWER_SIZE - 1u;

// MITM split for the low22 layer: 12 high bits + 10 low bits
static constexpr int MITM_LOW_BITS = 10;
static constexpr int MITM_HIGH_BITS = LOWER_BITS - MITM_LOW_BITS; // 12
static constexpr uint32_t MITM_LOW_SIZE = 1u << MITM_LOW_BITS;    // 1024
static constexpr uint32_t MITM_HIGH_SIZE = 1u << MITM_HIGH_BITS;  // 4096
static constexpr uint32_t MITM_LOW_MASK = MITM_LOW_SIZE - 1u;
static constexpr uint32_t MITM_HIGH_MASK = MITM_HIGH_SIZE - 1u;

void handleSigint(int) {
    stopRequested.store(true, memory_order_relaxed);
}

inline uint32_t computeRegionSeedFromBase(uint32_t base) noexcept {
    return base ^ (DW::FEATURE_KEY + (base << 6) + (base >> 2) - 1640531527u);
}

// Fast MT19937
static constexpr uint32_t MT_A = 1812433253u;
static constexpr uint32_t TWIST_B = 0x9908b0dfu;

inline uint32_t temper(uint32_t y) noexcept {
    y ^= (y >> 11);
    y ^= (y << 7) & 2636928640u;
    y ^= (y << 15) & 4022730752u;
    y ^= (y >> 18);
    return y;
}

inline uint32_t twistOnce(uint32_t a, uint32_t b, uint32_t c) noexcept {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    uint32_t m = c ^ (y >> 1);
    if (y & 1u) m ^= TWIST_B;
    return m;
}

inline DW::FeatureSeed MakeFeatureSeed(uint32_t seedLow32) noexcept {
    uint32_t mt0 = seedLow32;
    uint32_t mt1 = 0, mt2 = 0, mt397 = 0, mt398 = 0;

    uint32_t prev = seedLow32;
    for (uint32_t i = 1; i <= 398; ++i) {
        prev = MT_A * (prev ^ (prev >> 30)) + i;
        if (i == 1) mt1 = prev;
        else if (i == 2) mt2 = prev;
        else if (i == 397) mt397 = prev;
        else if (i == 398) mt398 = prev;
    }

    uint32_t raw0 = temper(twistOnce(mt0, mt1, mt397));
    uint32_t raw1 = temper(twistOnce(mt1, mt2, mt398));

    return DW::FeatureSeed{
        seedLow32,
        (raw0 >> 1) | 1u,
        (raw1 >> 1) | 1u,
        DW::FEATURE_KEY
    };
}

inline void printProgress(const string &phaseName, uint64_t processed, uint64_t total, double rate) {
    lock_guard<mutex> guard(outputMutex);
    double percent = total ? (100.0 * processed / double(total)) : 0.0;
    cerr << '[' << phaseName << "] "
         << fixed << setprecision(3) << percent << "% (" << processed << '/' << total << ')'
         << " rate=" << rate << "/s";
    if (phaseName == "search") {
        cerr << " found=" << foundCount.load(memory_order_relaxed)
             << " candidates=" << candidateCount.load(memory_order_relaxed);
    }
    cerr << '\n';
}

bool solveLinearDiophantine(int64_t a, int64_t b, int64_t c,
                            int64_t &x0, int64_t &y0, int64_t &g) {
    auto extgcd = [&](auto self, int64_t aa, int64_t bb) -> pair<int64_t, int64_t> {
        if (bb == 0) return {1, 0};
        auto p = self(self, bb, aa % bb);
        __int128 x = p.second;
        __int128 y = (__int128)p.first - (__int128)(aa / bb) * p.second;
        return { (int64_t)x, (int64_t)y };
    };

    g = gcd(a, b);
    if (g < 0) g = -g;
    if (g == 0) return c == 0;
    if (c % g != 0) return false;

    auto uv = extgcd(extgcd, a, b);
    __int128 scale = (__int128)c / g;
    x0 = (int64_t)((__int128)uv.first * scale);
    y0 = (int64_t)((__int128)uv.second * scale);
    return true;
}

struct BestSol {
    uint32_t seed;
    uint32_t xMul;
    uint32_t zMul;
    uint32_t baseO;
    int32_t chunkX;
    int32_t chunkZ;
    uint64_t distance;
};

static inline __int128 iabs128(__int128 v) noexcept {
    return v < 0 ? -v : v;
}

static inline __int128 floorDiv128(__int128 a, __int128 b) noexcept {
    __int128 q = a / b;
    __int128 r = a % b;
    if (r != 0 && a < 0) --q;
    return q;
}

static inline __int128 ceilDiv128(__int128 a, __int128 b) noexcept {
    __int128 q = a / b;
    __int128 r = a % b;
    if (r != 0 && a > 0) ++q;
    return q;
}

BestSol nearest(uint32_t seed, uint32_t xMul, uint32_t zMul, uint32_t baseO) {
    int64_t cx0 = 0;
    int64_t cz0 = 0;
    int64_t g = 0;

    if (!solveLinearDiophantine(int64_t(xMul), int64_t(zMul), int64_t(int32_t(baseO)), cx0, cz0, g)) {
        return BestSol{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }
    if (g <= 0) {
        return BestSol{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }

    int64_t stepX = int64_t(zMul) / g;
    int64_t stepZ = int64_t(xMul) / g;

    __int128 ax0 = cx0;
    __int128 az0 = cz0;
    __int128 sx = stepX;
    __int128 sz = stepZ;

    auto feasible = [&](uint64_t D, __int128 &loK, __int128 &hiK) -> bool {
        __int128 d = (__int128)D;
        __int128 lo1 = ceilDiv128(-d - ax0, sx);
        __int128 hi1 = floorDiv128(d - ax0, sx);
        __int128 lo2 = ceilDiv128(az0 - d, sz);
        __int128 hi2 = floorDiv128(az0 + d, sz);
        loK = max(lo1, lo2);
        hiK = min(hi1, hi2);
        return loK <= hiK;
    };

    uint64_t upper = uint64_t(max(iabs128(ax0), iabs128(az0)));
    uint64_t lo = 0, hi = upper;

    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        __int128 kLo = 0, kHi = 0;
        if (feasible(mid, kLo, kHi)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    __int128 kLo = 0, kHi = 0;
    if (!feasible(lo, kLo, kHi)) {
        return BestSol{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }

    __int128 k = kLo;
    __int128 bestCx = ax0 + k * sx;
    __int128 bestCz = az0 - k * sz;

    return BestSol{
        seed, xMul, zMul, baseO, int32_t(bestCx), int32_t(bestCz), lo
    };
}

struct Phase {
    string name;
    uint64_t total;
    atomic<uint64_t> processed{};
};

void reporter(const Phase &status) {
    uint64_t lastProcessed = 0;
    auto lastTime = steady_clock::now();

    while (!stopRequested.load(memory_order_relaxed) &&
           status.processed.load(memory_order_relaxed) < status.total) {
        this_thread::sleep_for(seconds(REPORT_INTERVAL_SECONDS));
        if (stopRequested.load(memory_order_relaxed)) break;

        auto now = steady_clock::now();
        double elapsed = duration<double>(now - lastTime).count();
        uint64_t current = status.processed.load(memory_order_relaxed);
        double rate = elapsed > 0.0 ? double(current - lastProcessed) / elapsed : 0.0;
        printProgress(status.name, current, status.total, rate);
        lastTime = now;
        lastProcessed = current;
    }

    uint64_t current = status.processed.load(memory_order_relaxed);
    printProgress(status.name, current, status.total, 0.0);
}

// SIMD Optimization using AVX2
struct alignas(64) RM1024 {
    uint64_t w[32]{};
};

struct alignas(32) M1024 {
    __m256i v[4];

    inline void loadRot(const uint64_t* w, unsigned wordShift, __m128i shiftR, __m128i shiftL, bool doShift) noexcept {
        const uint64_t* src = w + wordShift;
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(src + 0));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(src + 4));
        __m256i a2 = _mm256_loadu_si256((const __m256i*)(src + 8));
        __m256i a3 = _mm256_loadu_si256((const __m256i*)(src + 12));

        if (doShift) {
            __m256i b0 = _mm256_loadu_si256((const __m256i*)(src + 1));
            __m256i b1 = _mm256_loadu_si256((const __m256i*)(src + 5));
            __m256i b2 = _mm256_loadu_si256((const __m256i*)(src + 9));
            __m256i b3 = _mm256_loadu_si256((const __m256i*)(src + 13));

            v[0] = _mm256_or_si256(_mm256_srl_epi64(a0, shiftR), _mm256_sll_epi64(b0, shiftL));
            v[1] = _mm256_or_si256(_mm256_srl_epi64(a1, shiftR), _mm256_sll_epi64(b1, shiftL));
            v[2] = _mm256_or_si256(_mm256_srl_epi64(a2, shiftR), _mm256_sll_epi64(b2, shiftL));
            v[3] = _mm256_or_si256(_mm256_srl_epi64(a3, shiftR), _mm256_sll_epi64(b3, shiftL));
        } else {
            v[0] = a0; v[1] = a1; v[2] = a2; v[3] = a3;
        }
    }

    inline void bitAnd(const M1024& other) noexcept {
        v[0] = _mm256_and_si256(v[0], other.v[0]);
        v[1] = _mm256_and_si256(v[1], other.v[1]);
        v[2] = _mm256_and_si256(v[2], other.v[2]);
        v[3] = _mm256_and_si256(v[3], other.v[3]);
    }

    inline void bitOr(const M1024& other) noexcept {
        v[0] = _mm256_or_si256(v[0], other.v[0]);
        v[1] = _mm256_or_si256(v[1], other.v[1]);
        v[2] = _mm256_or_si256(v[2], other.v[2]);
        v[3] = _mm256_or_si256(v[3], other.v[3]);
    }

    inline bool any() const noexcept {
        __m256i o01 = _mm256_or_si256(v[0], v[1]);
        __m256i o23 = _mm256_or_si256(v[2], v[3]);
        __m256i final_or = _mm256_or_si256(o01, o23);
        return !_mm256_testz_si256(final_or, final_or);
    }
};

struct ShiftParams {
    unsigned wordShift;
    unsigned bitShift;
    unsigned bitShift2;
};

static inline __m256i swapBits256(__m256i v, __m256i mask, int shift) noexcept {
    __m256i s = _mm256_srli_epi64(v, shift);
    __m256i t = _mm256_and_si256(_mm256_xor_si256(s, v), mask);
    return _mm256_xor_si256(_mm256_xor_si256(v, t), _mm256_slli_epi64(t, shift));
}

static inline void xorPerm(RM1024 &m, uint32_t x) noexcept {
    __m256i v0 = _mm256_loadu_si256((__m256i*)&m.w[0]);
    __m256i v1 = _mm256_loadu_si256((__m256i*)&m.w[4]);
    __m256i v2 = _mm256_loadu_si256((__m256i*)&m.w[8]);
    __m256i v3 = _mm256_loadu_si256((__m256i*)&m.w[12]);

    if (x & 1u) {
        __m256i mask = _mm256_set1_epi64x(0x5555555555555555ull);
        v0 = swapBits256(v0, mask, 1); v1 = swapBits256(v1, mask, 1);
        v2 = swapBits256(v2, mask, 1); v3 = swapBits256(v3, mask, 1);
    }
    if (x & 2u) {
        __m256i mask = _mm256_set1_epi64x(0x3333333333333333ull);
        v0 = swapBits256(v0, mask, 2); v1 = swapBits256(v1, mask, 2);
        v2 = swapBits256(v2, mask, 2); v3 = swapBits256(v3, mask, 2);
    }
    if (x & 4u) {
        __m256i mask = _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0full);
        v0 = swapBits256(v0, mask, 4); v1 = swapBits256(v1, mask, 4);
        v2 = swapBits256(v2, mask, 4); v3 = swapBits256(v3, mask, 4);
    }
    if (x & 8u) {
        __m256i mask = _mm256_set1_epi64x(0x00ff00ff00ff00ffull);
        v0 = swapBits256(v0, mask, 8); v1 = swapBits256(v1, mask, 8);
        v2 = swapBits256(v2, mask, 8); v3 = swapBits256(v3, mask, 8);
    }
    if (x & 16u) {
        __m256i mask = _mm256_set1_epi64x(0x0000ffff0000ffffull);
        v0 = swapBits256(v0, mask, 16); v1 = swapBits256(v1, mask, 16);
        v2 = swapBits256(v2, mask, 16); v3 = swapBits256(v3, mask, 16);
    }
    if (x & 32u) {
        __m256i mask = _mm256_set1_epi64x(0x00000000ffffffffull);
        v0 = swapBits256(v0, mask, 32); v1 = swapBits256(v1, mask, 32);
        v2 = swapBits256(v2, mask, 32); v3 = swapBits256(v3, mask, 32);
    }

    if (x & 64u) {
        v0 = _mm256_permute4x64_epi64(v0, 0xB1); v1 = _mm256_permute4x64_epi64(v1, 0xB1);
        v2 = _mm256_permute4x64_epi64(v2, 0xB1); v3 = _mm256_permute4x64_epi64(v3, 0xB1);
    }
    if (x & 128u) {
        v0 = _mm256_permute4x64_epi64(v0, 0x4E); v1 = _mm256_permute4x64_epi64(v1, 0x4E);
        v2 = _mm256_permute4x64_epi64(v2, 0x4E); v3 = _mm256_permute4x64_epi64(v3, 0x4E);
    }
    if (x & 256u) {
        swap(v0, v1); swap(v2, v3);
    }
    if (x & 512u) {
        swap(v0, v2); swap(v1, v3);
    }

    // Mirror store for the ring buffer wrapping offset!
    _mm256_storeu_si256((__m256i*)&m.w[0], v0);
    _mm256_storeu_si256((__m256i*)&m.w[16], v0);
    
    _mm256_storeu_si256((__m256i*)&m.w[4], v1);
    _mm256_storeu_si256((__m256i*)&m.w[20], v1);
    
    _mm256_storeu_si256((__m256i*)&m.w[8], v2);
    _mm256_storeu_si256((__m256i*)&m.w[24], v2);
    
    _mm256_storeu_si256((__m256i*)&m.w[12], v3);
    _mm256_storeu_si256((__m256i*)&m.w[28], v3);
}

static inline RM1024 mkPrefixMask(uint32_t bits) noexcept {
    RM1024 m{};
    if (bits == 0) return m;
    uint32_t fullWords = bits / 64u;
    uint32_t rem = bits % 64u;
    for (uint32_t i = 0; i < fullWords; ++i) m.w[i] = ~0ull;
    if (rem != 0u && fullWords < 16u) m.w[fullWords] = (1ull << rem) - 1ull;
    return m;
}

static array<RM1024, MITM_LOW_SIZE> cm0;
static array<RM1024, MITM_LOW_SIZE> cm1;
static array<M1024, MITM_LOW_SIZE> avxMask0;
static array<M1024, MITM_LOW_SIZE> avxMask1;

static void initMasks() {
    for (uint32_t d = 0; d < MITM_LOW_SIZE; ++d) {
        cm0[d] = mkPrefixMask(MITM_LOW_SIZE - d);
        for (int i = 0; i < 16; ++i) {
            cm1[d].w[i] = ~cm0[d].w[i];
        }
        
        avxMask0[d].v[0] = _mm256_loadu_si256((__m256i*)&cm0[d].w[0]);
        avxMask0[d].v[1] = _mm256_loadu_si256((__m256i*)&cm0[d].w[4]);
        avxMask0[d].v[2] = _mm256_loadu_si256((__m256i*)&cm0[d].w[8]);
        avxMask0[d].v[3] = _mm256_loadu_si256((__m256i*)&cm0[d].w[12]);

        avxMask1[d].v[0] = _mm256_loadu_si256((__m256i*)&cm1[d].w[0]);
        avxMask1[d].v[1] = _mm256_loadu_si256((__m256i*)&cm1[d].w[4]);
        avxMask1[d].v[2] = _mm256_loadu_si256((__m256i*)&cm1[d].w[8]);
        avxMask1[d].v[3] = _mm256_loadu_si256((__m256i*)&cm1[d].w[12]);
    }
}

// Fast Lookup Bitsets and Data Arrays
static vector<uint64_t> swBitset;
static vector<uint64_t> neBitset;
static vector<uint64_t> nwBitset;
static vector<uint32_t> seBucketStart;
static vector<uint16_t> seBucketCount;
static vector<uint32_t> seValuesFlat;

static array<vector<uint32_t>, CORNER_COUNT> cSets;
static array<vector<RM1024>, CORNER_COUNT> cRows;

static void buildArtifacts(int corner) {
    auto &set = cSets[corner];
    auto &rows = cRows[corner];
    rows.assign(MITM_HIGH_SIZE, RM1024{});

    vector<uint32_t> sortedByLow22 = set;
    sort(sortedByLow22.begin(), sortedByLow22.end(), [](uint32_t a, uint32_t b) {
        uint32_t lowA = a & LOWER_MASK;
        uint32_t lowB = b & LOWER_MASK;
        if (lowA != lowB) return lowA < lowB;
        return a < b;
    });

    uint32_t prevLow22 = UINT32_MAX;
    for (uint32_t v : sortedByLow22) {
        uint32_t low22 = v & LOWER_MASK;
        if (low22 != prevLow22) {
            uint32_t row = low22 >> MITM_LOW_BITS;
            uint32_t bit = low22 & MITM_LOW_MASK;
            rows[row].w[bit >> 6] |= 1ull << (bit & 63u);
            prevLow22 = low22;
        }
    }

    if (corner == SW) {
        swBitset.assign(TOTAL_UINT32 / 64, 0);
        for (uint32_t v : set) swBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == NE) {
        neBitset.assign(TOTAL_UINT32 / 64, 0);
        for (uint32_t v : set) neBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == NW) {
        nwBitset.assign(TOTAL_UINT32 / 64, 0);
        for (uint32_t v : set) nwBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == SE) {
        seBucketStart.assign(LOWER_SIZE, 0);
        seBucketCount.assign(LOWER_SIZE, 0);
        seValuesFlat = sortedByLow22; 
        
        for (size_t i = 0; i < seValuesFlat.size(); ) {
            uint32_t low22 = seValuesFlat[i] & LOWER_MASK;
            size_t j = i;
            while (j < seValuesFlat.size() && (seValuesFlat[j] & LOWER_MASK) == low22) ++j;
            seBucketStart[low22] = uint32_t(i);
            seBucketCount[low22] = uint16_t(j - i);
            i = j;
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;
    uint64_t limitTotal = 1ull << 32;

    signal(SIGINT, handleSigint);
    initMasks();

    vector<vector<uint32_t>> vBases(CORNER_COUNT);
    Phase baseStatus{"Scanning all bases...", limitTotal};

    auto baseWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);

        array<vector<uint32_t>, CORNER_COUNT> localLists{};
        uint64_t localProcessed = 0;

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32_t base = uint32_t(value);
            uint32_t regionSeed = computeRegionSeedFromBase(base);
            DW::RNG rng(regionSeed);

            if (rng.next<500>() != 0u) {
                ++localProcessed;
                if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                    baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                    localProcessed = 0;
                }
                continue;
            }

            uint32_t offZ = rng.next<16>();
            uint32_t offX = rng.next<16>();

            auto inLowRange = [](uint32_t v) noexcept { return v <= 4u; };
            auto inHighRange = [](uint32_t v) noexcept { return v >= 11u && v <= 15u; };

            bool xLow = inLowRange(offX);
            bool xHigh = inHighRange(offX);
            bool zLow = inLowRange(offZ);
            bool zHigh = inHighRange(offZ);

            for (int corner = 0; corner < CORNER_COUNT; ++corner) {
                bool match = false;
                switch (corner) {
                    case NW: match = xLow && zLow; break;
                    case NE: match = xHigh && zLow; break;
                    case SW: match = xLow && zHigh; break;
                    case SE: match = xHigh && zHigh; break;
                }
                if (match) {
                    localLists[corner].push_back(base);
                    break;
                }
            }

            ++localProcessed;
            if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                localProcessed = 0;
            }
        }

        baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
        lock_guard<mutex> guard(outputMutex);
        for (int corner = 0; corner < CORNER_COUNT; ++corner) {
            auto &globalList = vBases[corner];
            auto &localList = localLists[corner];
            globalList.insert(globalList.end(), localList.begin(), localList.end());
        }
    };

    vector<thread> baseThreads;
    baseThreads.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t) {
        baseThreads.emplace_back(baseWorker, t);
    }
    thread baseReporter(reporter, cref(baseStatus));

    for (auto &t : baseThreads) t.join();
    stopRequested.store(false, memory_order_relaxed);
    if (baseReporter.joinable()) baseReporter.join();

    cerr << "Phase 1 complete\n";

    for (int corner = 0; corner < CORNER_COUNT; ++corner) {
        auto &list = vBases[corner];
        sort(list.begin(), list.end());
        list.erase(unique(list.begin(), list.end()), list.end());
        cSets[corner] = move(list);

        buildArtifacts(corner);

        if (cSets[corner].empty()) {
            cerr << "ERROR: no candidates for corner " << corner << "\n";
            return 1;
        }
    }

    Phase searchStatus{"Searching for solutions...", limitTotal};
    BestSol best{0, 0, 0, 0, 0, 0, UINT64_MAX};
    mutex bestMutex;

    auto searchWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        blockSize = (blockSize + 1023u) & ~1023ULL; 
        uint64_t start = min(uint64_t(tid) * blockSize, limitTotal);
        uint64_t end = min(start + blockSize, limitTotal);

        uint64_t localProcessed = 0;
        BestSol localBest{0, 0, 0, 0, 0, 0, UINT64_MAX};

        auto sePre = make_unique<RM1024[]>(MITM_HIGH_SIZE);
        auto swPre = make_unique<RM1024[]>(MITM_HIGH_SIZE);
        auto nePre = make_unique<RM1024[]>(MITM_HIGH_SIZE);
        auto nwPre = make_unique<RM1024[]>(MITM_HIGH_SIZE);

        const auto *seRows = cRows[SE].data();
        const auto *swRows = cRows[SW].data();
        const auto *neRows = cRows[NE].data();
        const auto *nwRows = cRows[NW].data();

        for (uint32_t sl10 = 0; sl10 < 1024; ++sl10) {
            if (stopRequested.load(memory_order_relaxed)) break;

            for (uint32_t i = 0; i < MITM_HIGH_SIZE; ++i) {
                sePre[i] = seRows[i]; xorPerm(sePre[i], sl10);
                swPre[i] = swRows[i]; xorPerm(swPre[i], sl10);
                nePre[i] = neRows[i]; xorPerm(nePre[i], sl10);
                nwPre[i] = nwRows[i]; xorPerm(nwPre[i], sl10);
            }

            for (uint64_t base = start; base < end; base += 1024) {
                if (stopRequested.load(memory_order_relaxed)) break;
                uint64_t currentSeedVal = base + sl10;
                if (currentSeedVal >= end) continue;

                uint32_t seed = uint32_t(currentSeedVal);
                auto feat = MakeFeatureSeed(seed);

                uint32_t xMul = feat.xMul;
                uint32_t zMul = feat.zMul;

                uint32_t seedLow22 = seed & LOWER_MASK;
                uint32_t seedHigh12 = (seedLow22 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

                uint32_t xLow22 = xMul & LOWER_MASK;
                uint32_t zLow22 = zMul & LOWER_MASK;
                uint32_t xzLow22 = (xLow22 + zLow22) & LOWER_MASK;

                uint32_t xLow10 = xLow22 & MITM_LOW_MASK;
                uint32_t zLow10 = zLow22 & MITM_LOW_MASK;
                uint32_t xzLow10 = xzLow22 & MITM_LOW_MASK;

                uint32_t xHigh12 = (xLow22 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                uint32_t zHigh12 = (zLow22 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                uint32_t xzHigh12 = (xzLow22 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

                auto getShiftParams = [](unsigned r) -> ShiftParams {
                    r &= 1023u;
                    return { r >> 6, r & 63u, 64u - (r & 63u) };
                };
                
                ShiftParams spSW = getShiftParams(xLow10);
                ShiftParams spNE = getShiftParams(zLow10);
                ShiftParams spNW = getShiftParams(xzLow10);
                
                bool doSW = xLow10 != 0u;
                bool doNE = zLow10 != 0u;
                bool doNW = xzLow10 != 0u;

                bool doShiftSW = spSW.bitShift != 0;
                bool doShiftNE = spNE.bitShift != 0;
                bool doShiftNW = spNW.bitShift != 0;

                __m128i shiftR_SW = _mm_cvtsi32_si128(spSW.bitShift);
                __m128i shiftL_SW = _mm_cvtsi32_si128(spSW.bitShift2);
                
                __m128i shiftR_NE = _mm_cvtsi32_si128(spNE.bitShift);
                __m128i shiftL_NE = _mm_cvtsi32_si128(spNE.bitShift2);
                
                __m128i shiftR_NW = _mm_cvtsi32_si128(spNW.bitShift);
                __m128i shiftL_NW = _mm_cvtsi32_si128(spNW.bitShift2);

                const M1024& carry0_SW = avxMask0[xLow10];
                const M1024& carry1_SW = avxMask1[xLow10];
                
                const M1024& carry0_NE = avxMask0[zLow10];
                const M1024& carry1_NE = avxMask1[zLow10];
                
                const M1024& carry0_NW = avxMask0[xzLow10];
                const M1024& carry1_NW = avxMask1[xzLow10];

                for (uint32_t row0 = 0; row0 < MITM_HIGH_SIZE; ++row0) {
                    uint32_t uHigh12 = seedHigh12 ^ row0;
                    
                    M1024 mask;
                    mask.v[0] = _mm256_loadu_si256((__m256i*)&sePre[row0].w[0]);
                    mask.v[1] = _mm256_loadu_si256((__m256i*)&sePre[row0].w[4]);
                    mask.v[2] = _mm256_loadu_si256((__m256i*)&sePre[row0].w[8]);
                    mask.v[3] = _mm256_loadu_si256((__m256i*)&sePre[row0].w[12]);

                    // SW Phase
                    uint32_t sumSW = (uHigh12 + xHigh12) & MITM_HIGH_MASK;
                    uint32_t rowSW = seedHigh12 ^ sumSW;
                    
                    M1024 extraSW;
                    extraSW.loadRot(swPre[rowSW].w, spSW.wordShift, shiftR_SW, shiftL_SW, doShiftSW);
                    extraSW.bitAnd(carry0_SW);

                    if (doSW) {
                        uint32_t rowSW1 = seedHigh12 ^ ((sumSW + 1u) & MITM_HIGH_MASK);
                        M1024 extraSW1;
                        extraSW1.loadRot(swPre[rowSW1].w, spSW.wordShift, shiftR_SW, shiftL_SW, doShiftSW);
                        extraSW1.bitAnd(carry1_SW);
                        extraSW.bitOr(extraSW1);
                    }
                    mask.bitAnd(extraSW);
                    if (!mask.any()) continue;

                    // NE Phase
                    uint32_t sumNE = (uHigh12 + zHigh12) & MITM_HIGH_MASK;
                    uint32_t rowNE = seedHigh12 ^ sumNE;
                    
                    M1024 extraNE;
                    extraNE.loadRot(nePre[rowNE].w, spNE.wordShift, shiftR_NE, shiftL_NE, doShiftNE);
                    extraNE.bitAnd(carry0_NE);

                    if (doNE) {
                        uint32_t rowNE1 = seedHigh12 ^ ((sumNE + 1u) & MITM_HIGH_MASK);
                        M1024 extraNE1;
                        extraNE1.loadRot(nePre[rowNE1].w, spNE.wordShift, shiftR_NE, shiftL_NE, doShiftNE);
                        extraNE1.bitAnd(carry1_NE);
                        extraNE.bitOr(extraNE1);
                    }
                    mask.bitAnd(extraNE);
                    if (!mask.any()) continue;

                    // NW Phase
                    uint32_t sumNW = (uHigh12 + xzHigh12) & MITM_HIGH_MASK;
                    uint32_t rowNW = seedHigh12 ^ sumNW;
                    
                    M1024 extraNW;
                    extraNW.loadRot(nwPre[rowNW].w, spNW.wordShift, shiftR_NW, shiftL_NW, doShiftNW);
                    extraNW.bitAnd(carry0_NW);

                    if (doNW) {
                        uint32_t rowNW1 = seedHigh12 ^ ((sumNW + 1u) & MITM_HIGH_MASK);
                        M1024 extraNW1;
                        extraNW1.loadRot(nwPre[rowNW1].w, spNW.wordShift, shiftR_NW, shiftL_NW, doShiftNW);
                        extraNW1.bitAnd(carry1_NW);
                        extraNW.bitOr(extraNW1);
                    }
                    mask.bitAnd(extraNW);
                    if (!mask.any()) continue;

                    alignas(32) uint64_t maskWords[16];
                    _mm256_storeu_si256((__m256i*)&maskWords[0], mask.v[0]);
                    _mm256_storeu_si256((__m256i*)&maskWords[4], mask.v[1]);
                    _mm256_storeu_si256((__m256i*)&maskWords[8], mask.v[2]);
                    _mm256_storeu_si256((__m256i*)&maskWords[12], mask.v[3]);

                    for (int word = 0; word < 16; ++word) {
                        uint64_t bits = maskWords[word];
                        while (bits) {
                            unsigned bit = unsigned(__builtin_ctzll(bits));
                            bits &= (bits - 1);

                            uint32_t uLow10 = uint32_t(word * 64 + bit);
                            uint32_t u24 = (uHigh12 << MITM_LOW_BITS) | uLow10;
                            uint32_t baseSELow22 = seedLow22 ^ u24;
                            
                            uint32_t startIdx = seBucketStart[baseSELow22];
                            uint32_t count = seBucketCount[baseSELow22];

                            for (uint32_t k = 0; k < count; ++k) {
                                if ((k & 1023u) == 0u && stopRequested.load(memory_order_relaxed)) break;

                                uint32_t baseSE = seValuesFlat[startIdx + k];
                                uint32_t cSE = seed ^ baseSE;

                                uint32_t baseSW = seed ^ (cSE + xMul);
                                if (!((swBitset[baseSW >> 6] >> (baseSW & 63)) & 1)) continue;

                                uint32_t baseNE = seed ^ (cSE + zMul);
                                if (!((neBitset[baseNE >> 6] >> (baseNE & 63)) & 1)) continue;

                                uint32_t baseNW = seed ^ (cSE + xMul + zMul);
                                if (!((nwBitset[baseNW >> 6] >> (baseNW & 63)) & 1)) continue;

                                candidateCount.fetch_add(1, memory_order_relaxed);
                                foundCount.fetch_add(1, memory_order_relaxed);

                                uint32_t baseO = cSE;
                                BestSol solution = nearest(seed, xMul, zMul, baseO);
                                if (solution.distance < localBest.distance) {
                                    localBest = solution;
                                }
                            }
                        }
                    }
                }

                ++localProcessed;
                if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                    searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                    localProcessed = 0;
                }
            }
        }
        
        searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);

        if (localBest.distance < UINT64_MAX) {
            lock_guard<mutex> guard(bestMutex);
            if (localBest.distance < best.distance) {
                best = localBest;
            }
        }
    };

    vector<thread> searchThreads;
    searchThreads.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t) {
        searchThreads.emplace_back(searchWorker, t);
    }
    thread searchReporter(reporter, cref(searchStatus));

    for (auto &t : searchThreads) t.join();
    stopRequested.store(true, memory_order_relaxed);
    if (searchReporter.joinable()) searchReporter.join();

    if (best.distance < UINT64_MAX) {
        cerr << "FOUND seed=" << best.seed
             << " xMul=" << best.xMul
             << " zMul=" << best.zMul
             << " originChunk=(" << best.chunkX << ',' << best.chunkZ << ')'
             << " distance=" << best.distance << '\n';
    } else {
        cerr << "No valid solution found in scanned range.\n";
    }

    int dummy;
    cin >> dummy; //prevents window from closing
    return 0;
}