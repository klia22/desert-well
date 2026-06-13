#include <bits/stdc++.h>
#include "well.h" //Used only for RNG constants
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
using namespace std::chrono;

#define uint32_t uint32;
#define uint64_t uint64;
#define __int128 int128;

static atomic<bool> stopRequested{false};
static atomic<uint64> foundCount{0};
static atomic<uint64> candidateCount{0};
static mutex outputMutex;

enum Corner : int {
    NW = 0,
    NE = 1,
    SW = 2,
    SE = 3,
    CORNER_COUNT = 4
};

static constexpr uint64 TOTAL_Uuint32 = 0x1'0000'0000ULL;
static constexpr uint32 REPORT_INTERVAL_SECONDS = 20;
static constexpr uint64 FLUSH_INTERVAL = 20;

// Exact low22 filtering (most optimal tracking lower bit amount)
static constexpr int LOWER_BITS = 22;
static constexpr uint32 LOWER_SIZE = 1u << LOWER_BITS;
static constexpr uint32 LOWER_MASK = LOWER_SIZE - 1u;

// MITM split for the low22 layer: 12 high bits + 10 low bits (most optimial split after testing)
static constexpr int MITM_LOW_BITS = 10;
static constexpr int MITM_HIGH_BITS = LOWER_BITS - MITM_LOW_BITS; // 12
static constexpr uint32 MITM_LOW_SIZE = 1u << MITM_LOW_BITS;    // 1024
static constexpr uint32 MITM_HIGH_SIZE = 1u << MITM_HIGH_BITS;  // 4096
static constexpr uint32 MITM_LOW_MASK = MITM_LOW_SIZE - 1u;
static constexpr uint32 MITM_HIGH_MASK = MITM_HIGH_SIZE - 1u;

void handleSigint(int) {
    stopRequested.store(true, memory_order_relaxed);
}

inline uint32 computeRegionSeedFromBase(uint32 base) noexcept {
    return base ^ (DW::FEATURE_KEY + (base << 6) + (base >> 2) - 1640531527u);
}

// Fast MT19937
static constexpr uint32 MT_A = 1812433253u;
static constexpr uint32 TWIST_B = 0x9908b0dfu;

inline uint32 temper(uint32 y) noexcept {
    y ^= (y >> 11);
    y ^= (y << 7) & 2636928640u;
    y ^= (y << 15) & 4022730752u;
    y ^= (y >> 18);
    return y;
}

inline uint32 twistOnce(uint32 a, uint32 b, uint32 c) noexcept {
    uint32 y = (a & 0x80000000u) | (b & 0x7fffffffu);
    uint32 m = c ^ (y >> 1);
    if (y & 1u) m ^= TWIST_B;
    return m;
}

inline DW::FeatureSeed fastMakeFeatureSeed(uint32 seedLow32) noexcept {
    uint32 mt0 = seedLow32;
    uint32 mt1 = 0, mt2 = 0, mt397 = 0, mt398 = 0;

    uint32 prev = seedLow32;
    for (uint32 i = 1; i <= 398; ++i) {
        prev = MT_A * (prev ^ (prev >> 30)) + i;
        if (i == 1) mt1 = prev;
        else if (i == 2) mt2 = prev;
        else if (i == 397) mt397 = prev;
        else if (i == 398) mt398 = prev;
    }

    uint32 raw0 = temper(twistOnce(mt0, mt1, mt397));
    uint32 raw1 = temper(twistOnce(mt1, mt2, mt398));

    return DW::FeatureSeed{
        seedLow32,
        (raw0 >> 1) | 1u,
        (raw1 >> 1) | 1u,
        DW::FEATURE_KEY
    };
}

inline void printProgress(const string &phaseName, uint64 processed, uint64 total, double rate) {
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

bool solveLinearDiophantine(uint64_t a, uint64_t b, uint64_t c,
                            uint64_t &x0, uint64_t &y0, uint64_t &g) {
    auto extgcd = [&](auto self, uint64_t aa, uint64_t bb) -> pair<uint64_t, uint64_t> {
        if (bb == 0) return {1, 0};
        auto p = self(self, bb, aa % bb);
        int128 x = p.second;
        int128 y = (int128)p.first - (int128)(aa / bb) * p.second;
        return { (uint64_t)x, (uint64_t)y };
    };

    g = std::gcd(a, b);
    if (g < 0) g = -g;
    if (g == 0) return c == 0;
    if (c % g != 0) return false;

    auto uv = extgcd(extgcd, a, b);
    int128 scale = (int128)c / g;
    x0 = (uint64_t)((int128)uv.first * scale);
    y0 = (uint64_t)((int128)uv.second * scale);
    return true;
}

struct BestSolution {
    uint32 seed;
    uint32 xMul;
    uint32 zMul;
    uint32 baseO;
    uint32_t chunkX;
    uint32_t chunkZ;
    uint64 distance;
};

static inline int128 iabs128(int128 v) noexcept {
    return v < 0 ? -v : v;
}

static inline int128 floorDiv128(int128 a, int128 b) noexcept {
    int128 q = a / b;
    int128 r = a % b;
    if (r != 0 && a < 0) --q;
    return q;
}

static inline int128 ceilDiv128(int128 a, int128 b) noexcept {
    int128 q = a / b;
    int128 r = a % b;
    if (r != 0 && a > 0) ++q;
    return q;
}

BestSolution nearestSolution(uint32 seed, uint32 xMul, uint32 zMul, uint32 baseO) {
    uint64_t cx0 = 0;
    uint64_t cz0 = 0;
    uint64_t g = 0;

    if (!solveLinearDiophantine(uint64_t(xMul), uint64_t(zMul), uint64_t(uint32_t(baseO)), cx0, cz0, g)) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, Uuint64_MAX};
    }
    if (g <= 0) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, Uuint64_MAX};
    }

    uint64_t stepX = uint64_t(zMul) / g;
    uint64_t stepZ = uint64_t(xMul) / g;

    int128 ax0 = cx0;
    int128 az0 = cz0;
    int128 sx = stepX;
    int128 sz = stepZ;

    auto feasible = [&](uint64 D, int128 &loK, int128 &hiK) -> bool {
        int128 d = (int128)D;
        int128 lo1 = ceilDiv128(-d - ax0, sx);
        int128 hi1 = floorDiv128(d - ax0, sx);
        int128 lo2 = ceilDiv128(az0 - d, sz);
        int128 hi2 = floorDiv128(az0 + d, sz);
        loK = max(lo1, lo2);
        hiK = min(hi1, hi2);
        return loK <= hiK;
    };

    uint64 upper = uint64(max(iabs128(ax0), iabs128(az0)));
    uint64 lo = 0, hi = upper;

    while (lo < hi) {
        uint64 mid = lo + ((hi - lo) >> 1);
        int128 kLo = 0, kHi = 0;
        if (feasible(mid, kLo, kHi)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    int128 kLo = 0, kHi = 0;
    if (!feasible(lo, kLo, kHi)) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, Uuint64_MAX};
    }

    int128 k = kLo;
    int128 bestCx = ax0 + k * sx;
    int128 bestCz = az0 - k * sz;

    return BestSolution{
        seed, xMul, zMul, baseO, uint32_t(bestCx), uint32_t(bestCz), lo
    };
}

struct PhaseStatus {
    string name;
    uint64 total;
    atomic<uint64> processed{};
};

void reporterThread(const PhaseStatus &status) {
    uint64 lastProcessed = 0;
    auto lastTime = steady_clock::now();

    while (!stopRequested.load(memory_order_relaxed) &&
           status.processed.load(memory_order_relaxed) < status.total) {
        this_thread::sleep_for(seconds(REPORT_INTERVAL_SECONDS));
        if (stopRequested.load(memory_order_relaxed)) break;

        auto now = steady_clock::now();
        double elapsed = duration<double>(now - lastTime).count();
        uint64 current = status.processed.load(memory_order_relaxed);
        double rate = elapsed > 0.0 ? double(current - lastProcessed) / elapsed : 0.0;
        printProgress(status.name, current, status.total, rate);
        lastTime = now;
        lastProcessed = current;
    }

    uint64 current = status.processed.load(memory_order_relaxed);
    printProgress(status.name, current, status.total, 0.0);
}

// SIMD Optimization 
struct alignas(64) RowMask1024 {
    uint64 w[32]{};
};

struct alignas(32) Mask1024AVX2 {
    __m256i v[4];

    inline void load_rotated(const uint64* w, unsigned wordShift, __m128i shiftR, __m128i shiftL, bool doShift) noexcept {
        const uint64* src = w + wordShift;
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

    inline void bitwise_and(const Mask1024AVX2& other) noexcept {
        v[0] = _mm256_and_si256(v[0], other.v[0]);
        v[1] = _mm256_and_si256(v[1], other.v[1]);
        v[2] = _mm256_and_si256(v[2], other.v[2]);
        v[3] = _mm256_and_si256(v[3], other.v[3]);
    }

    inline void bitwise_or(const Mask1024AVX2& other) noexcept {
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

static inline void xorPermute1024AVX2(RowMask1024 &m, uint32 x) noexcept {
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
        std::swap(v0, v1); std::swap(v2, v3);
    }
    if (x & 512u) {
        std::swap(v0, v2); std::swap(v1, v3);
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

static inline RowMask1024 makePrefixMask1024(uint32 bits) noexcept {
    RowMask1024 m{};
    if (bits == 0) return m;
    uint32 fullWords = bits / 64u;
    uint32 rem = bits % 64u;
    for (uint32 i = 0; i < fullWords; ++i) m.w[i] = ~0ull;
    if (rem != 0u && fullWords < 16u) m.w[fullWords] = (1ull << rem) - 1ull;
    return m;
}

static array<RowMask1024, MITM_LOW_SIZE> carryMask0;
static array<RowMask1024, MITM_LOW_SIZE> carryMask1;
static array<Mask1024AVX2, MITM_LOW_SIZE> avxCarryMask0;
static array<Mask1024AVX2, MITM_LOW_SIZE> avxCarryMask1;

static void initCarryMasks() {
    for (uint32 d = 0; d < MITM_LOW_SIZE; ++d) {
        carryMask0[d] = makePrefixMask1024(MITM_LOW_SIZE - d);
        for (int i = 0; i < 16; ++i) {
            carryMask1[d].w[i] = ~carryMask0[d].w[i];
        }
        
        avxCarryMask0[d].v[0] = _mm256_loadu_si256((__m256i*)&carryMask0[d].w[0]);
        avxCarryMask0[d].v[1] = _mm256_loadu_si256((__m256i*)&carryMask0[d].w[4]);
        avxCarryMask0[d].v[2] = _mm256_loadu_si256((__m256i*)&carryMask0[d].w[8]);
        avxCarryMask0[d].v[3] = _mm256_loadu_si256((__m256i*)&carryMask0[d].w[12]);

        avxCarryMask1[d].v[0] = _mm256_loadu_si256((__m256i*)&carryMask1[d].w[0]);
        avxCarryMask1[d].v[1] = _mm256_loadu_si256((__m256i*)&carryMask1[d].w[4]);
        avxCarryMask1[d].v[2] = _mm256_loadu_si256((__m256i*)&carryMask1[d].w[8]);
        avxCarryMask1[d].v[3] = _mm256_loadu_si256((__m256i*)&carryMask1[d].w[12]);
    }
}

// Fast Lookup Bitsets and Data Arrays
static vector<uint64> swBitset;
static vector<uint64> neBitset;
static vector<uint64> nwBitset;
static vector<uint32> seBucketStart;
static vector<uint16_t> seBucketCount;
static vector<uint32> seValuesFlat;

static array<vector<uint32>, CORNER_COUNT> cornerSets;
static array<vector<RowMask1024>, CORNER_COUNT> cornerRows;

static void buildCornerArtifacts(int corner) {
    auto &set = cornerSets[corner];
    auto &rows = cornerRows[corner];
    rows.assign(MITM_HIGH_SIZE, RowMask1024{});

    vector<uint32> sortedByLow22 = set;
    sort(sortedByLow22.begin(), sortedByLow22.end(), [](uint32 a, uint32 b) {
        uint32 lowA = a & LOWER_MASK;
        uint32 lowB = b & LOWER_MASK;
        if (lowA != lowB) return lowA < lowB;
        return a < b;
    });

    uint32 prevLow22 = Uuint32_MAX;
    for (uint32 v : sortedByLow22) {
        uint32 low22 = v & LOWER_MASK;
        if (low22 != prevLow22) {
            uint32 row = low22 >> MITM_LOW_BITS;
            uint32 bit = low22 & MITM_LOW_MASK;
            rows[row].w[bit >> 6] |= 1ull << (bit & 63u);
            prevLow22 = low22;
        }
    }

    if (corner == SW) {
        swBitset.assign(TOTAL_Uuint32 / 64, 0);
        for (uint32 v : set) swBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == NE) {
        neBitset.assign(TOTAL_Uuint32 / 64, 0);
        for (uint32 v : set) neBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == NW) {
        nwBitset.assign(TOTAL_Uuint32 / 64, 0);
        for (uint32 v : set) nwBitset[v >> 6] |= (1ULL << (v & 63u));
    } else if (corner == SE) {
        seBucketStart.assign(LOWER_SIZE, 0);
        seBucketCount.assign(LOWER_SIZE, 0);
        seValuesFlat = sortedByLow22; 
        
        for (size_t i = 0; i < seValuesFlat.size(); ) {
            uint32 low22 = seValuesFlat[i] & LOWER_MASK;
            size_t j = i;
            while (j < seValuesFlat.size() && (seValuesFlat[j] & LOWER_MASK) == low22) ++j;
            seBucketStart[low22] = uint32(i);
            seBucketCount[low22] = uint16_t(j - i);
            i = j;
        }
    }
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;
    uint64 limitTotal = 1ull << 32;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threadCount = unsigned(stoi(argv[++i]));
        } else if (arg == "--limit" && i + 1 < argc) {
            limitTotal = stoull(argv[++i]);
            if (limitTotal == 0) limitTotal = TOTAL_Uuint32;
        }
    }

    signal(SIGINT, handleSigint);
    initCarryMasks();

    vector<vector<uint32>> validBases(CORNER_COUNT);
    PhaseStatus baseStatus{"base-scan", limitTotal};

    auto baseWorker = [&](unsigned tid) {
        uint64 blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64 start = uint64(tid) * blockSize;
        uint64 end = min(start + blockSize, limitTotal);

        array<vector<uint32>, CORNER_COUNT> localLists{};
        uint64 localProcessed = 0;

        for (uint64 value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32 base = uint32(value);
            uint32 regionSeed = computeRegionSeedFromBase(base);
            DW::RandWrapper rng(regionSeed);

            if (rng.nextInt<500>() != 0u) {
                ++localProcessed;
                if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                    baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                    localProcessed = 0;
                }
                continue;
            }

            uint32 offZ = rng.nextInt<16>();
            uint32 offX = rng.nextInt<16>();

            auto inLowRange = [](uint32 v) noexcept { return v <= 3u; };
            auto inHighRange = [](uint32 v) noexcept { return v >= 14u && v <= 15u; };

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
            auto &globalList = validBases[corner];
            auto &localList = localLists[corner];
            globalList.insert(globalList.end(), localList.begin(), localList.end());
        }
    };

    vector<thread> baseThreads;
    baseThreads.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t) {
        baseThreads.emplace_back(baseWorker, t);
    }
    thread baseReporter(reporterThread, cref(baseStatus));

    for (auto &t : baseThreads) t.join();
    stopRequested.store(false, memory_order_relaxed);
    if (baseReporter.joinable()) baseReporter.join();

    cerr << "base phase complete\n";

    for (int corner = 0; corner < CORNER_COUNT; ++corner) {
        auto &list = validBases[corner];
        sort(list.begin(), list.end());
        list.erase(unique(list.begin(), list.end()), list.end());
        cornerSets[corner] = move(list);

        buildCornerArtifacts(corner);

        cerr << "corner " << corner << " candidates=" << cornerSets[corner].size() << "\n";
        if (cornerSets[corner].empty()) {
            cerr << "ERROR: no candidates for corner " << corner << "\n";
            return 1;
        }
    }

    PhaseStatus searchStatus{"search", limitTotal};
    BestSolution best{0, 0, 0, 0, 0, 0, Uuint64_MAX};
    mutex bestMutex;

    auto searchWorker = [&](unsigned tid) {
        uint64 blockSize = (limitTotal + threadCount - 1) / threadCount;
        blockSize = (blockSize + 1023u) & ~1023ULL; 
        uint64 start = min(uint64(tid) * blockSize, limitTotal);
        uint64 end = min(start + blockSize, limitTotal);

        uint64 localProcessed = 0;
        BestSolution localBest{0, 0, 0, 0, 0, 0, Uuint64_MAX};

        auto pre_seRows = make_unique<RowMask1024[]>(MITM_HIGH_SIZE);
        auto pre_swRows = make_unique<RowMask1024[]>(MITM_HIGH_SIZE);
        auto pre_neRows = make_unique<RowMask1024[]>(MITM_HIGH_SIZE);
        auto pre_nwRows = make_unique<RowMask1024[]>(MITM_HIGH_SIZE);

        const auto *seRows = cornerRows[SE].data();
        const auto *swRows = cornerRows[SW].data();
        const auto *neRows = cornerRows[NE].data();
        const auto *nwRows = cornerRows[NW].data();

        for (uint32 sl10 = 0; sl10 < 1024; ++sl10) {
            if (stopRequested.load(memory_order_relaxed)) break;

            for (uint32 i = 0; i < MITM_HIGH_SIZE; ++i) {
                pre_seRows[i] = seRows[i]; xorPermute1024AVX2(pre_seRows[i], sl10);
                pre_swRows[i] = swRows[i]; xorPermute1024AVX2(pre_swRows[i], sl10);
                pre_neRows[i] = neRows[i]; xorPermute1024AVX2(pre_neRows[i], sl10);
                pre_nwRows[i] = nwRows[i]; xorPermute1024AVX2(pre_nwRows[i], sl10);
            }

            for (uint64 base = start; base < end; base += 1024) {
                if (stopRequested.load(memory_order_relaxed)) break;
                uint64 currentSeedVal = base + sl10;
                if (currentSeedVal >= end) continue;

                uint32 seed = uint32(currentSeedVal);
                auto feat = fastMakeFeatureSeed(seed);

                uint32 xMul = feat.xMul;
                uint32 zMul = feat.zMul;

                uint32 seedLow24 = seed & LOWER_MASK;
                uint32 seedHigh14 = (seedLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

                uint32 xLow24 = xMul & LOWER_MASK;
                uint32 zLow24 = zMul & LOWER_MASK;
                uint32 xzLow24 = (xLow24 + zLow24) & LOWER_MASK;

                uint32 xLow10 = xLow24 & MITM_LOW_MASK;
                uint32 zLow10 = zLow24 & MITM_LOW_MASK;
                uint32 xzLow10 = xzLow24 & MITM_LOW_MASK;

                uint32 xHigh14 = (xLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                uint32 zHigh14 = (zLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                uint32 xzHigh14 = (xzLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

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

                const Mask1024AVX2& carry0_SW = avxCarryMask0[xLow10];
                const Mask1024AVX2& carry1_SW = avxCarryMask1[xLow10];
                
                const Mask1024AVX2& carry0_NE = avxCarryMask0[zLow10];
                const Mask1024AVX2& carry1_NE = avxCarryMask1[zLow10];
                
                const Mask1024AVX2& carry0_NW = avxCarryMask0[xzLow10];
                const Mask1024AVX2& carry1_NW = avxCarryMask1[xzLow10];

                for (uint32 row0 = 0; row0 < MITM_HIGH_SIZE; ++row0) {
                    uint32 uHigh14 = seedHigh14 ^ row0;
                    
                    Mask1024AVX2 mask;
                    mask.v[0] = _mm256_loadu_si256((__m256i*)&pre_seRows[row0].w[0]);
                    mask.v[1] = _mm256_loadu_si256((__m256i*)&pre_seRows[row0].w[4]);
                    mask.v[2] = _mm256_loadu_si256((__m256i*)&pre_seRows[row0].w[8]);
                    mask.v[3] = _mm256_loadu_si256((__m256i*)&pre_seRows[row0].w[12]);

                    // SW Phase
                    uint32 sumSW = (uHigh14 + xHigh14) & MITM_HIGH_MASK;
                    uint32 rowSW = seedHigh14 ^ sumSW;
                    
                    Mask1024AVX2 extraSW;
                    extraSW.load_rotated(pre_swRows[rowSW].w, spSW.wordShift, shiftR_SW, shiftL_SW, doShiftSW);
                    extraSW.bitwise_and(carry0_SW);

                    if (doSW) {
                        uint32 rowSW1 = seedHigh14 ^ ((sumSW + 1u) & MITM_HIGH_MASK);
                        Mask1024AVX2 extraSW1;
                        extraSW1.load_rotated(pre_swRows[rowSW1].w, spSW.wordShift, shiftR_SW, shiftL_SW, doShiftSW);
                        extraSW1.bitwise_and(carry1_SW);
                        extraSW.bitwise_or(extraSW1);
                    }
                    mask.bitwise_and(extraSW);
                    if (!mask.any()) continue;

                    // NE Phase
                    uint32 sumNE = (uHigh14 + zHigh14) & MITM_HIGH_MASK;
                    uint32 rowNE = seedHigh14 ^ sumNE;
                    
                    Mask1024AVX2 extraNE;
                    extraNE.load_rotated(pre_neRows[rowNE].w, spNE.wordShift, shiftR_NE, shiftL_NE, doShiftNE);
                    extraNE.bitwise_and(carry0_NE);

                    if (doNE) {
                        uint32 rowNE1 = seedHigh14 ^ ((sumNE + 1u) & MITM_HIGH_MASK);
                        Mask1024AVX2 extraNE1;
                        extraNE1.load_rotated(pre_neRows[rowNE1].w, spNE.wordShift, shiftR_NE, shiftL_NE, doShiftNE);
                        extraNE1.bitwise_and(carry1_NE);
                        extraNE.bitwise_or(extraNE1);
                    }
                    mask.bitwise_and(extraNE);
                    if (!mask.any()) continue;

                    // NW Phase
                    uint32 sumNW = (uHigh14 + xzHigh14) & MITM_HIGH_MASK;
                    uint32 rowNW = seedHigh14 ^ sumNW;
                    
                    Mask1024AVX2 extraNW;
                    extraNW.load_rotated(pre_nwRows[rowNW].w, spNW.wordShift, shiftR_NW, shiftL_NW, doShiftNW);
                    extraNW.bitwise_and(carry0_NW);

                    if (doNW) {
                        uint32 rowNW1 = seedHigh14 ^ ((sumNW + 1u) & MITM_HIGH_MASK);
                        Mask1024AVX2 extraNW1;
                        extraNW1.load_rotated(pre_nwRows[rowNW1].w, spNW.wordShift, shiftR_NW, shiftL_NW, doShiftNW);
                        extraNW1.bitwise_and(carry1_NW);
                        extraNW.bitwise_or(extraNW1);
                    }
                    mask.bitwise_and(extraNW);
                    if (!mask.any()) continue;

                    alignas(32) uint64 maskWords[16];
                    _mm256_storeu_si256((__m256i*)&maskWords[0], mask.v[0]);
                    _mm256_storeu_si256((__m256i*)&maskWords[4], mask.v[1]);
                    _mm256_storeu_si256((__m256i*)&maskWords[8], mask.v[2]);
                    _mm256_storeu_si256((__m256i*)&maskWords[12], mask.v[3]);

                    for (int word = 0; word < 16; ++word) {
                        uint64 bits = maskWords[word];
                        while (bits) {
                            unsigned bit = unsigned(__builtin_ctzll(bits));
                            bits &= (bits - 1);

                            uint32 uLow10 = uint32(word * 64 + bit);
                            uint32 u24 = (uHigh14 << MITM_LOW_BITS) | uLow10;
                            uint32 baseSELow24 = seedLow24 ^ u24;
                            
                            uint32 startIdx = seBucketStart[baseSELow24];
                            uint32 count = seBucketCount[baseSELow24];

                            for (uint32 k = 0; k < count; ++k) {
                                if ((k & 1023u) == 0u && stopRequested.load(memory_order_relaxed)) break;

                                uint32 baseSE = seValuesFlat[startIdx + k];
                                uint32 cSE = seed ^ baseSE;

                                uint32 baseSW = seed ^ (cSE + xMul);
                                if (!((swBitset[baseSW >> 6] >> (baseSW & 63)) & 1)) continue;

                                uint32 baseNE = seed ^ (cSE + zMul);
                                if (!((neBitset[baseNE >> 6] >> (baseNE & 63)) & 1)) continue;

                                uint32 baseNW = seed ^ (cSE + xMul + zMul);
                                if (!((nwBitset[baseNW >> 6] >> (baseNW & 63)) & 1)) continue;

                                candidateCount.fetch_add(1, memory_order_relaxed);
                                foundCount.fetch_add(1, memory_order_relaxed);

                                uint32 baseO = cSE;
                                BestSolution solution = nearestSolution(seed, xMul, zMul, baseO);
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

        if (localBest.distance < Uuint64_MAX) {
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
    thread searchReporter(reporterThread, cref(searchStatus));

    for (auto &t : searchThreads) t.join();
    stopRequested.store(true, memory_order_relaxed);
    if (searchReporter.joinable()) searchReporter.join();

    if (best.distance < Uuint64_MAX) {
        cerr << "FOUND seed=" << best.seed
             << " xMul=" << best.xMul
             << " zMul=" << best.zMul
             << " originChunk=(" << best.chunkX << ',' << best.chunkZ << ')'
             << " distance=" << best.distance << '\n';
    } else {
        cerr << "No valid solution found in scanned range.\n";
    }

    int a;
    cin >> a; //prevents window from closing
    return 0;
}
