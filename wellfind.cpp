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
static constexpr uint32_t REPORT_INTERVAL_SECONDS = 1;
static constexpr uint64_t FLUSH_INTERVAL = 1ull << 16;
static constexpr int LOWER_BITS = 20;
static constexpr uint32_t LOWER_SIZE = 1u << LOWER_BITS;
static constexpr uint32_t LOWER_MASK = LOWER_SIZE - 1u;

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

struct BestSolution {
    uint32_t seed;
    uint32_t xMul;
    uint32_t zMul;
    uint32_t baseO;
    int32_t chunkX;
    int32_t chunkZ;
    uint64_t distanceSquared;
};

inline uint64_t squaredDistance(int32_t x, int32_t z) noexcept {
    int64_t xx = int64_t(x);
    int64_t zz = int64_t(z);
    return uint64_t(xx * xx + zz * zz);
}

bool solveLinearDiophantine(int64_t a, int64_t b, int64_t c, int64_t &x0, int64_t &y0, int64_t &g) {
    auto extgcd = [&](auto self, int64_t aa, int64_t bb) -> pair<int64_t, int64_t> {
        if (bb == 0) return {1, 0};
        auto p = self(self, bb, aa % bb);
        return pair{p.second, p.first - (aa / bb) * p.second};
    };
    auto uv = extgcd(extgcd, a, b);
    g = a * uv.first + b * uv.second;
    if (g < 0) {
        g = -g;
        uv.first = -uv.first;
        uv.second = -uv.second;
    }
    if (c % g != 0) return false;
    x0 = uv.first * (c / g);
    y0 = uv.second * (c / g);
    return true;
}

BestSolution nearestSolution(uint32_t seed, uint32_t xMul, uint32_t zMul, uint32_t baseO) {
    int64_t cx0 = 0;
    int64_t cz0 = 0;
    int64_t g = 0;
    if (!solveLinearDiophantine(int64_t(xMul), int64_t(zMul), int64_t(int32_t(baseO)), cx0, cz0, g)) {
        return BestSolution{seed, xMul, zMul, baseO, 0, 0, UINT64_MAX};
    }
    int64_t stepX = int64_t(zMul) / g;
    int64_t stepZ = int64_t(xMul) / g;
    long double ideal = -static_cast<long double>(cx0) / static_cast<long double>(stepX);
    int64_t bestK = llround(ideal);
    int64_t bestCx = cx0 + bestK * stepX;
    int64_t bestCz = cz0 - bestK * stepZ;
    uint64_t bestDistance = squaredDistance(int32_t(bestCx), int32_t(bestCz));
    for (int delta = -2; delta <= 2; ++delta) {
        int64_t candidateK = bestK + delta;
        int64_t candidateCx = cx0 + candidateK * stepX;
        int64_t candidateCz = cz0 - candidateK * stepZ;
        uint64_t candidateDistance = squaredDistance(int32_t(candidateCx), int32_t(candidateCz));
        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;
            bestCx = candidateCx;
            bestCz = candidateCz;
        }
    }
    return BestSolution{seed, xMul, zMul, baseO, int32_t(bestCx), int32_t(bestCz), bestDistance};
}

struct PhaseStatus {
    string name;
    uint64_t total;
    atomic<uint64_t> processed{};
};

void reporterThread(const PhaseStatus &status) {
    uint64_t lastProcessed = 0;
    auto lastTime = steady_clock::now();
    while (!stopRequested.load(memory_order_relaxed) && status.processed.load(memory_order_relaxed) < status.total) {
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

struct BucketTable {
    array<uint32_t, LOWER_SIZE> begin{};
    array<uint32_t, LOWER_SIZE> count{};
    array<uint8_t, LOWER_SIZE> present{};
    vector<uint32_t> keys;
    vector<uint32_t> values;

    inline bool contains(uint32_t value) const noexcept {
        uint32_t low = value & LOWER_MASK;
        if (!present[low]) return false;
        uint32_t b = begin[low];
        uint32_t e = b + count[low];
        return binary_search(values.begin() + b, values.begin() + e, value);
    }
};

static inline uint32_t low20(uint32_t v) noexcept {
    return v & LOWER_MASK;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;
    uint64_t limitTotal = 1ull << 22;

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
    for (unsigned t = 0; t < threadCount; ++t) {
        baseThreads.emplace_back(baseWorker, t);
    }
    thread baseReporter(reporterThread, cref(baseStatus));

    for (auto &t : baseThreads) t.join();
    stopRequested.store(false, memory_order_relaxed);
    if (baseReporter.joinable()) baseReporter.join();
    cerr << "base phase complete\n";

    array<vector<uint32_t>, CORNER_COUNT> cornerSets;
    array<BucketTable, CORNER_COUNT> tables;

    for (int corner = 0; corner < CORNER_COUNT; ++corner) {
        auto &list = validBases[corner];
        sort(list.begin(), list.end());
        list.erase(unique(list.begin(), list.end()), list.end());
        cornerSets[corner] = move(list);

        auto &tbl = tables[corner];
        vector<pair<uint32_t, uint32_t>> items;
        items.reserve(cornerSets[corner].size());
        for (uint32_t v : cornerSets[corner]) {
            items.push_back({v & LOWER_MASK, v});
        }

        sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        for (size_t i = 0; i < items.size(); ) {
            uint32_t low = items[i].first;
            tbl.present[low] = 1;
            tbl.begin[low] = uint32_t(tbl.values.size());
            tbl.keys.push_back(low);
            size_t j = i;
            while (j < items.size() && items[j].first == low) {
                tbl.values.push_back(items[j].second);
                ++j;
            }
            tbl.count[low] = uint32_t(j - i);
            i = j;
        }

        cerr << "corner " << corner << " candidates=" << cornerSets[corner].size() << "\n";
        if (cornerSets[corner].empty()) {
            cerr << "ERROR: no candidates for corner " << corner << "\n";
            return 1;
        }
    }

    PhaseStatus searchStatus{"search", limitTotal};
    BestSolution best{0, 0, 0, 0, 0, 0, UINT64_MAX};
    mutex bestMutex;

    int iterCorner = SE;
    if (tables[SW].keys.size() < tables[iterCorner].keys.size()) iterCorner = SW;
    if (tables[NE].keys.size() < tables[iterCorner].keys.size()) iterCorner = NE;
    if (tables[NW].keys.size() < tables[iterCorner].keys.size()) iterCorner = NW;

    auto searchWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);
        uint64_t localProcessed = 0;
        BestSolution localBest{0, 0, 0, 0, 0, 0, UINT64_MAX};

        const auto &seTbl = tables[SE];
        const auto &swTbl = tables[SW];
        const auto &neTbl = tables[NE];
        const auto &nwTbl = tables[NW];
        const auto &iterKeys = tables[iterCorner].keys;

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32_t seed = uint32_t(value);
            auto feat = fastMakeFeatureSeed(seed);
            uint32_t xMul = feat.xMul;
            uint32_t zMul = feat.zMul;
            uint32_t seedLow = seed & LOWER_MASK;
            uint32_t xLo = xMul & LOWER_MASK;
            uint32_t zLo = zMul & LOWER_MASK;
            uint32_t xzLo = (xLo + zLo) & LOWER_MASK;

            auto toSELow = [&](int corner, uint32_t k) -> uint32_t {
                switch (corner) {
                    case SE: return k;
                    case SW: return (seedLow ^ ((seedLow ^ k) - xLo)) & LOWER_MASK;
                    case NE: return (seedLow ^ ((seedLow ^ k) - zLo)) & LOWER_MASK;
                    case NW: return (seedLow ^ ((seedLow ^ k) - xzLo)) & LOWER_MASK;
                    default: return k;
                }
            };

            auto fromSELow = [&](uint32_t baseSELow, uint32_t deltaLow) -> uint32_t {
                return (seedLow ^ ((seedLow ^ baseSELow) + deltaLow)) & LOWER_MASK;
            };

            bool anyFoundLow = false;
            for (uint32_t k : iterKeys) {
                uint32_t candidateSE_low = toSELow(iterCorner, k);
                uint32_t swLow = fromSELow(candidateSE_low, xLo);
                uint32_t neLow = fromSELow(candidateSE_low, zLo);
                uint32_t nwLow = fromSELow(candidateSE_low, xzLo);

                if (!seTbl.present[candidateSE_low]) continue;
                if (!swTbl.present[swLow]) continue;
                if (!neTbl.present[neLow]) continue;
                if (!nwTbl.present[nwLow]) continue;

                uint32_t b = seTbl.begin[candidateSE_low];
                uint32_t e = b + seTbl.count[candidateSE_low];

                for (uint32_t idx = b; idx < e; ++idx) {
                    uint32_t baseSE = seTbl.values[idx];
                    uint32_t cSE = seed ^ baseSE;

                    uint32_t baseSW = seed ^ (cSE + xMul);
                    uint32_t baseNE = seed ^ (cSE + zMul);
                    uint32_t baseNW = seed ^ (cSE + xMul + zMul);

                    if (!swTbl.contains(baseSW)) continue;
                    if (!neTbl.contains(baseNE)) continue;
                    if (!nwTbl.contains(baseNW)) continue;

                    anyFoundLow = true;
                    candidateCount.fetch_add(1, memory_order_relaxed);
                    foundCount.fetch_add(1, memory_order_relaxed);

                    uint32_t baseO = cSE;
                    BestSolution solution = nearestSolution(seed, xMul, zMul, baseO);
                    if (solution.distanceSquared < localBest.distanceSquared) {
                        localBest = solution;
                    }
                    break;
                }

                if (localBest.distanceSquared < UINT64_MAX) break;
            }

            (void)anyFoundLow;
            ++localProcessed;
            if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                localProcessed = 0;
            }
        }

        searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
        if (localBest.distanceSquared < UINT64_MAX) {
            lock_guard<mutex> guard(bestMutex);
            if (localBest.distanceSquared < best.distanceSquared) {
                best = localBest;
            }
        }
    };

    vector<thread> searchThreads;
    for (unsigned t = 0; t < threadCount; ++t) {
        searchThreads.emplace_back(searchWorker, t);
    }
    thread searchReporter(reporterThread, cref(searchStatus));

    for (auto &t : searchThreads) t.join();
    stopRequested.store(true, memory_order_relaxed);
    if (searchReporter.joinable()) searchReporter.join();

    if (best.distanceSquared < UINT64_MAX) {
        cerr << "FOUND seed=" << best.seed
             << " xMul=" << best.xMul
             << " zMul=" << best.zMul
             << " originChunk=(" << best.chunkX << ',' << best.chunkZ << ')'
             << " distance^2=" << best.distanceSquared << '\n';
    } else {
        cerr << "No valid solution found in scanned range.\n";
    }

    int a;
    cin >> a;
    return 0;
}
