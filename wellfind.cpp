#include <bits/stdc++.h>
#include "well.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <numeric>

using namespace std;
using namespace std::chrono;

static atomic<bool> stopRequested{false};
static atomic<uint64_t> foundCount{0};
static atomic<uint64_t> candidateCount{0};
static mutex outputMutex;

enum Corner : int {
    NW = 0,
    NE = 1,
    SW = 2,
    SE = 3,
    CORNER_COUNT = 4
};

static constexpr uint64_t TOTAL_UINT32 = 0x1'0000'0000ULL;
static constexpr uint32_t REPORT_INTERVAL_SECONDS = 20;
static constexpr uint64_t FLUSH_INTERVAL = 1ull << 10;

// Exact low24 filtering
static constexpr int LOWER_BITS = 24;
static constexpr uint32_t LOWER_SIZE = 1u << LOWER_BITS;
static constexpr uint32_t LOWER_MASK = LOWER_SIZE - 1u;

// MITM split for the low24 layer: 14 high bits + 10 low bits
static constexpr int MITM_LOW_BITS = 10;
static constexpr int MITM_HIGH_BITS = LOWER_BITS - MITM_LOW_BITS; // 14
static constexpr uint32_t MITM_LOW_SIZE = 1u << MITM_LOW_BITS;    // 1024
static constexpr uint32_t MITM_HIGH_SIZE = 1u << MITM_HIGH_BITS;  // 16384
static constexpr uint32_t MITM_LOW_MASK = MITM_LOW_SIZE - 1u;
static constexpr uint32_t MITM_HIGH_MASK = MITM_HIGH_SIZE - 1u;

void handleSigint(int) {
    stopRequested.store(true, memory_order_relaxed);
}

inline uint32_t computeRegionSeedFromBase(uint32_t base) noexcept {
    return base ^ (DW::FEATURE_KEY + (base << 6) + (base >> 2) - 1640531527u);
}

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

inline DW::FeatureSeed fastMakeFeatureSeed(uint32_t seedLow32) noexcept {
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
    }; // xMul and zMul: upper 31 bits random, last bit is 1 (always odd)
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

    g = std::gcd(a, b);
    if (g < 0) g = -g;
    if (g == 0) return c == 0;

    if (c % g != 0) return false;

    auto uv = extgcd(extgcd, a, b);

    __int128 scale = (__int128)c / g;
    x0 = (int64_t)((__int128)uv.first * scale);
    y0 = (int64_t)((__int128)uv.second * scale);
    return true;
}

struct BestSolution {
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

BestSolution nearestSolution(uint32_t seed, uint32_t xMul, uint32_t zMul, uint32_t baseO) {
    int64_t cx0 = 0;
    int64_t cz0 = 0;
    int64_t g = 0;

    if (!solveLinearDiophantine(int64_t(xMul), int64_t(zMul), int64_t(int32_t(baseO)), cx0, cz0, g)) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }
    if (g <= 0) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
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
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }

    __int128 k = kLo;
    __int128 bestCx = ax0 + k * sx;
    __int128 bestCz = az0 - k * sz;

    return BestSolution{
        seed,
        xMul,
        zMul,
        baseO,
        int32_t(bestCx),
        int32_t(bestCz),
        lo
    };
}

struct PhaseStatus {
    string name;
    uint64_t total;
    atomic<uint64_t> processed{};
};

void reporterThread(const PhaseStatus &status) {
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

struct alignas(64) RowMask1024 {
    uint64_t w[16]{};
};

static inline bool anyMask(const RowMask1024 &m) noexcept {
    for (uint64_t v : m.w) {
        if (v) return true;
    }
    return false;
}

static inline void andMask(RowMask1024 &a, const RowMask1024 &b) noexcept {
    for (int i = 0; i < 16; ++i) a.w[i] &= b.w[i];
}

static inline void orMask(RowMask1024 &a, const RowMask1024 &b) noexcept {
    for (int i = 0; i < 16; ++i) a.w[i] |= b.w[i];
}

static inline uint64_t swapBits64(uint64_t x, uint64_t mask, unsigned shift) noexcept {
    uint64_t t = ((x >> shift) ^ x) & mask;
    return x ^ t ^ (t << shift);
}

static inline void swapWordBlocks(RowMask1024 &m, unsigned blockWords) noexcept {
    const unsigned step = blockWords * 2u;
    for (unsigned base = 0; base < 16; base += step) {
        for (unsigned j = 0; j < blockWords; ++j) {
            std::swap(m.w[base + j], m.w[base + blockWords + j]);
        }
    }
}

static inline void xorPermute1024(RowMask1024 &m, uint32_t x) noexcept {
    if (x & 1u)  for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x5555555555555555ull, 1);
    if (x & 2u)  for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x3333333333333333ull, 2);
    if (x & 4u)  for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x0f0f0f0f0f0f0f0full, 4);
    if (x & 8u)  for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x00ff00ff00ff00ffull, 8);
    if (x & 16u) for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x0000ffff0000ffffull, 16);
    if (x & 32u) for (int i = 0; i < 16; ++i) m.w[i] = swapBits64(m.w[i], 0x00000000ffffffffull, 32);

    if (x & 64u)  swapWordBlocks(m, 1);
    if (x & 128u) swapWordBlocks(m, 2);
    if (x & 256u) swapWordBlocks(m, 4);
    if (x & 512u) swapWordBlocks(m, 8);
}

static inline RowMask1024 rotateRight1024(const RowMask1024 &in, unsigned r) noexcept {
    RowMask1024 out{};
    r &= 1023u;
    unsigned wordShift = r >> 6;
    unsigned bitShift = r & 63u;

    if (bitShift == 0u) {
        for (unsigned i = 0; i < 16; ++i) {
            out.w[i] = in.w[(i + wordShift) & 15u];
        }
        return out;
    }

    unsigned bitShift2 = 64u - bitShift;
    for (unsigned i = 0; i < 16; ++i) {
        uint64_t a = in.w[(i + wordShift) & 15u];
        uint64_t b = in.w[(i + wordShift + 1u) & 15u];
        out.w[i] = (a >> bitShift) | (b << bitShift2);
    }
    return out;
}

static inline RowMask1024 makePrefixMask1024(uint32_t bits) noexcept {
    RowMask1024 m{};
    if (bits == 0) return m;

    uint32_t fullWords = bits / 64u;
    uint32_t rem = bits % 64u;

    for (uint32_t i = 0; i < fullWords; ++i) {
        m.w[i] = ~0ull;
    }
    if (rem != 0u && fullWords < 16u) {
        m.w[fullWords] = (1ull << rem) - 1ull;
    }
    return m;
}

static array<RowMask1024, MITM_LOW_SIZE> carryMask0;
static array<RowMask1024, MITM_LOW_SIZE> carryMask1;

static void initCarryMasks() {
    for (uint32_t d = 0; d < MITM_LOW_SIZE; ++d) {
        carryMask0[d] = makePrefixMask1024(MITM_LOW_SIZE - d);
        for (int i = 0; i < 16; ++i) {
            carryMask1[d].w[i] = ~carryMask0[d].w[i];
        }
    }
}

struct BucketTable24 {
    static constexpr uint32_t BUCKET_SHIFT = MITM_LOW_BITS;   // 10
    static constexpr uint32_t BUCKET_SIZE = MITM_HIGH_SIZE;    // 16384

    array<uint32_t, BUCKET_SIZE> begin{};
    array<uint32_t, BUCKET_SIZE> count{};
    array<uint8_t, BUCKET_SIZE> present{};
    vector<uint32_t> low24Keys;
    vector<uint32_t> values;

    inline pair<uint32_t, uint32_t> exactRange(uint32_t low24) const noexcept {
        uint32_t bucket = low24 >> BUCKET_SHIFT;
        if (!present[bucket]) return {0, 0};

        uint32_t b = begin[bucket];
        uint32_t e = b + count[bucket];

        auto lb = lower_bound(low24Keys.begin() + b, low24Keys.begin() + e, low24);
        auto ub = upper_bound(lb, low24Keys.begin() + e, low24);

        return {
            uint32_t(lb - low24Keys.begin()),
            uint32_t(ub - low24Keys.begin())
        };
    }

    inline bool contains(uint32_t value) const noexcept {
        uint32_t low24 = value & LOWER_MASK;
        auto [lb, ub] = exactRange(low24);
        if (lb == ub) return false;
        return binary_search(values.begin() + lb, values.begin() + ub, value);
    }
};

static array<vector<uint32_t>, CORNER_COUNT> cornerSets;
static array<BucketTable24, CORNER_COUNT> tables;
static array<vector<RowMask1024>, CORNER_COUNT> cornerRows;

static void buildCornerArtifacts(int corner) {
    auto &set = cornerSets[corner];
    auto &tbl = tables[corner];
    auto &rows = cornerRows[corner];

    rows.assign(MITM_HIGH_SIZE, RowMask1024{});

    vector<pair<uint32_t, uint32_t>> items;
    items.reserve(set.size());

    for (uint32_t v : set) {
        items.emplace_back(v & LOWER_MASK, v);
    }

    sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    uint32_t prevLow24 = UINT32_MAX;
    for (const auto &it : items) {
        uint32_t low24 = it.first;
        if (low24 != prevLow24) {
            uint32_t row = low24 >> MITM_LOW_BITS;
            uint32_t bit = low24 & MITM_LOW_MASK;
            rows[row].w[bit >> 6] |= 1ull << (bit & 63u);
            prevLow24 = low24;
        }
    }

    tbl.begin.fill(0);
    tbl.count.fill(0);
    tbl.present.fill(0);
    tbl.low24Keys.clear();
    tbl.values.clear();
    tbl.low24Keys.reserve(items.size());
    tbl.values.reserve(items.size());

    size_t i = 0;
    while (i < items.size()) {
        uint32_t bucket = items[i].first >> BucketTable24::BUCKET_SHIFT;
        tbl.present[bucket] = 1;
        tbl.begin[bucket] = uint32_t(tbl.values.size());

        size_t j = i;
        while (j < items.size() && (items[j].first >> BucketTable24::BUCKET_SHIFT) == bucket) {
            tbl.low24Keys.push_back(items[j].first);
            tbl.values.push_back(items[j].second);
            ++j;
        }

        tbl.count[bucket] = uint32_t(j - i);
        i = j;
    }
}

static inline RowMask1024 buildCornerMask(
    const RowMask1024 *rows,
    uint32_t seedHigh14,
    uint32_t seedLow10,
    uint32_t uHigh14,
    uint32_t deltaHigh14,
    uint32_t deltaLow10
) noexcept {
    RowMask1024 out{};

    uint32_t sum0 = (uHigh14 + deltaHigh14) & MITM_HIGH_MASK;
    uint32_t row0 = seedHigh14 ^ sum0;

    out = rows[row0];
    xorPermute1024(out, seedLow10);
    out = rotateRight1024(out, deltaLow10);
    andMask(out, carryMask0[deltaLow10]);

    if (deltaLow10 != 0u) {
        uint32_t sum1 = (sum0 + 1u) & MITM_HIGH_MASK;
        uint32_t row1 = seedHigh14 ^ sum1;

        RowMask1024 extra = rows[row1];
        xorPermute1024(extra, seedLow10);
        extra = rotateRight1024(extra, deltaLow10);
        andMask(extra, carryMask1[deltaLow10]);
        orMask(out, extra);
    }

    return out;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;
    uint64_t limitTotal = 1ull << 32;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threadCount = unsigned(stoi(argv[++i]));
        } else if (arg == "--limit" && i + 1 < argc) {
            limitTotal = stoull(argv[++i]);
            if (limitTotal == 0) limitTotal = TOTAL_UINT32;
        }
    }

    signal(SIGINT, handleSigint);
    initCarryMasks();

    vector<vector<uint32_t>> validBases(CORNER_COUNT);
    PhaseStatus baseStatus{"base-scan", limitTotal};

    auto baseWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);

        array<vector<uint32_t>, CORNER_COUNT> localLists{};
        uint64_t localProcessed = 0;

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32_t base = uint32_t(value);
            uint32_t regionSeed = computeRegionSeedFromBase(base);
            DW::RandWrapper rng(regionSeed);

            if (rng.nextInt<500>() != 0u) {
                ++localProcessed;
                if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                    baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                    localProcessed = 0;
                }
                continue;
            }

            uint32_t offZ = rng.nextInt<16>();
            uint32_t offX = rng.nextInt<16>();

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
    BestSolution best{0, 0, 0, 0, 0, 0, UINT64_MAX};
    mutex bestMutex;

    auto searchWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);

        uint64_t localProcessed = 0;
        BestSolution localBest{0, 0, 0, 0, 0, 0, UINT64_MAX};

        const auto *seRows = cornerRows[SE].data();
        const auto *swRows = cornerRows[SW].data();
        const auto *neRows = cornerRows[NE].data();
        const auto *nwRows = cornerRows[NW].data();

        const auto &seTbl = tables[SE];
        const auto &swTbl = tables[SW];
        const auto &neTbl = tables[NE];
        const auto &nwTbl = tables[NW];

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32_t seed = uint32_t(value);
            auto feat = fastMakeFeatureSeed(seed);

            uint32_t xMul = feat.xMul;
            uint32_t zMul = feat.zMul;

            uint32_t seedLow24 = seed & LOWER_MASK;
            uint32_t seedLow10 = seedLow24 & MITM_LOW_MASK;
            uint32_t seedHigh14 = (seedLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

            uint32_t xLow24 = xMul & LOWER_MASK;
            uint32_t zLow24 = zMul & LOWER_MASK;
            uint32_t xzLow24 = (xLow24 + zLow24) & LOWER_MASK;

            uint32_t xLow10 = xLow24 & MITM_LOW_MASK;
            uint32_t zLow10 = zLow24 & MITM_LOW_MASK;
            uint32_t xzLow10 = xzLow24 & MITM_LOW_MASK;

            uint32_t xHigh14 = (xLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
            uint32_t zHigh14 = (zLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;
            uint32_t xzHigh14 = (xzLow24 >> MITM_LOW_BITS) & MITM_HIGH_MASK;

            for (uint32_t uHigh14 = 0; uHigh14 < MITM_HIGH_SIZE && !stopRequested.load(memory_order_relaxed); ++uHigh14) {
                RowMask1024 maskSE = buildCornerMask(seRows, seedHigh14, seedLow10, uHigh14, 0u, 0u);
                if (!anyMask(maskSE)) continue;

                RowMask1024 maskSW = buildCornerMask(swRows, seedHigh14, seedLow10, uHigh14, xHigh14, xLow10);
                if (!anyMask(maskSW)) continue;

                RowMask1024 maskNE = buildCornerMask(neRows, seedHigh14, seedLow10, uHigh14, zHigh14, zLow10);
                if (!anyMask(maskNE)) continue;

                RowMask1024 maskNW = buildCornerMask(nwRows, seedHigh14, seedLow10, uHigh14, xzHigh14, xzLow10);
                if (!anyMask(maskNW)) continue;

                andMask(maskSE, maskSW);
                if (!anyMask(maskSE)) continue;

                andMask(maskSE, maskNE);
                if (!anyMask(maskSE)) continue;

                andMask(maskSE, maskNW);
                if (!anyMask(maskSE)) continue;

                for (int word = 0; word < 16; ++word) {
                    uint64_t bits = maskSE.w[word];
                    while (bits) {
                        unsigned bit = unsigned(__builtin_ctzll(bits));
                        bits &= (bits - 1);

                        uint32_t uLow10 = uint32_t(word * 64 + bit);
                        uint32_t u24 = (uHigh14 << MITM_LOW_BITS) | uLow10;
                        uint32_t baseSELow24 = seedLow24 ^ u24;

                        auto [lb, ub] = seTbl.exactRange(baseSELow24);
                        if (lb == ub) continue;

                        for (uint32_t idx = lb; idx < ub; ++idx) {
                            if ((idx & 1023u) == 0u && stopRequested.load(memory_order_relaxed)) {
                                break;
                            }

                            uint32_t baseSE = seTbl.values[idx];
                            uint32_t cSE = seed ^ baseSE;

                            uint32_t baseSW = seed ^ (cSE + xMul);
                            uint32_t baseNE = seed ^ (cSE + zMul);
                            uint32_t baseNW = seed ^ (cSE + xMul + zMul);

                            if (!swTbl.contains(baseSW)) continue;
                            if (!neTbl.contains(baseNE)) continue;
                            if (!nwTbl.contains(baseNW)) continue;

                            candidateCount.fetch_add(1, memory_order_relaxed);
                            foundCount.fetch_add(1, memory_order_relaxed);

                            uint32_t baseO = cSE;
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
    thread searchReporter(reporterThread, cref(searchStatus));

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

    int a;
    cin >> a;
    return 0;
}