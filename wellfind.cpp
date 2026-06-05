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
static constexpr uint64_t FLUSH_INTERVAL = 1ull << 16; // 64k
// Use larger lower-bit hash buckets to speed up filtering (default 20 bits)
static constexpr int LOWER_BITS = 20;
static constexpr uint32_t LOWER_SIZE = 1u << LOWER_BITS;
static constexpr uint32_t LOWER_MASK = LOWER_SIZE - 1u;


static constexpr array<pair<int, int>, CORNER_COUNT> cornerOffsets = {
    pair{1, 1},   // NW
    pair{14, 1},  // NE
    pair{1, 14},  // SW
    pair{14, 14}  // SE
};

void handleSigint(int) {
    stopRequested.store(true, memory_order_relaxed);
}

inline uint32_t computeRegionSeedFromBase(uint32_t base) noexcept {
    return base ^ (DW::FEATURE_KEY + (base << 6) + (base >> 2) - 1640531527u);
}


inline int countTrailingZeros(uint64_t value) noexcept {
    return value ? __builtin_ctzll(value) : 64;
}

inline void printProgress(const string &phaseName, uint64_t processed, uint64_t total, double rate) {
    lock_guard<mutex> guard(outputMutex);
    double percent = total ? (100.0 * processed / double(total)) : 0.0;
    cerr << "[" << phaseName << "] "
         << fixed << setprecision(3) << percent << "% (" << processed << "/" << total << ")"
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
    // Print final state once when exiting (either complete or stopped)
    uint64_t current = status.processed.load(memory_order_relaxed);
    printProgress(status.name, current, status.total, 0.0);
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;
    // Default to searching ~2^22 seeds (user-specified optimization)
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
            for (int corner = 0; corner < CORNER_COUNT; ++corner) {
                // Accept a 5x5 corner region: x,z in 0..4 (low) or 11..15 (high)
                auto inLowRange = [](uint32_t v) noexcept { return v <= 4u; };
                auto inHighRange = [](uint32_t v) noexcept { return v >= 11u && v <= 15u; };
                bool xLow = inLowRange(offX);
                bool xHigh = inHighRange(offX);
                bool zLow = inLowRange(offZ);
                bool zHigh = inHighRange(offZ);
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
    array<unordered_map<uint32_t, vector<uint32_t>>, CORNER_COUNT> lowBuckets;
    array<vector<uint32_t>, CORNER_COUNT> lowKeys;

    for (int corner = 0; corner < CORNER_COUNT; ++corner) {
        auto &list = validBases[corner];
        sort(list.begin(), list.end());
        list.erase(unique(list.begin(), list.end()), list.end());
        cornerSets[corner] = move(list);
        for (uint32_t value : cornerSets[corner]) {
            uint32_t low = value & LOWER_MASK;
            lowBuckets[corner][low].push_back(value);
        }
        lowKeys[corner].reserve(lowBuckets[corner].size());
        for (auto &p : lowBuckets[corner]) lowKeys[corner].push_back(p.first);
        cerr << "corner " << corner << " candidates=" << cornerSets[corner].size() << "\n";
        if (cornerSets[corner].empty()) {
            cerr << "ERROR: no candidates for corner " << corner << "\n";
            return 1;
        }
    }

    PhaseStatus searchStatus{"search", limitTotal};
    BestSolution best{0, 0, 0, 0, 0, 0, UINT64_MAX};
    mutex bestMutex;
    

    auto isValidBase = [&](uint32_t baseSE, uint32_t xMul, uint32_t zMul) {
        uint32_t baseSW = baseSE ^ xMul;
        uint32_t baseNE = baseSE ^ zMul;
        uint32_t baseNW = baseSE ^ xMul ^ zMul;
        if (!binary_search(cornerSets[SW].begin(), cornerSets[SW].end(), baseSW)) return false;
        if (!binary_search(cornerSets[NE].begin(), cornerSets[NE].end(), baseNE)) return false;
        if (!binary_search(cornerSets[NW].begin(), cornerSets[NW].end(), baseNW)) return false;
        return true;
    };

    auto searchWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);
        uint64_t localProcessed = 0;
        BestSolution localBest{0, 0, 0, 0, 0, 0, UINT64_MAX};

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            uint32_t seed = uint32_t(value);
            auto feat = DW::makeFeatureSeed(int64_t(seed));
            uint32_t xMul = feat.xMul;
            uint32_t zMul = feat.zMul;
            uint32_t xLo = xMul & LOWER_MASK;
            uint32_t zLo = zMul & LOWER_MASK;
            uint32_t xzLo = xLo ^ zLo;

            // Choose the smallest low-key set among SE, SW, NE, NW to iterate
            size_t szSE = lowKeys[SE].size();
            size_t szSW = lowKeys[SW].size();
            size_t szNE = lowKeys[NE].size();
            size_t szNW = lowKeys[NW].size();
            int iterCorner = SE;
            size_t minSz = szSE;
            if (szSW < minSz) { minSz = szSW; iterCorner = SW; }
            if (szNE < minSz) { minSz = szNE; iterCorner = NE; }
            if (szNW < minSz) { minSz = szNW; iterCorner = NW; }

            bool anyFoundLow = false;
            const auto &iterKeys = lowKeys[iterCorner];
            for (uint32_t k : iterKeys) {
                uint32_t candidateSE_low;
                switch (iterCorner) {
                    case SE: candidateSE_low = k; break;
                    case SW: candidateSE_low = k ^ xLo; break;
                    case NE: candidateSE_low = k ^ zLo; break;
                    case NW: candidateSE_low = k ^ xzLo; break;
                    default: candidateSE_low = k; break;
                }
                // Check presence across all corners using low-buckets
                if (lowBuckets[SE].find(candidateSE_low) == lowBuckets[SE].end()) continue;
                if (lowBuckets[SW].find(candidateSE_low ^ xLo) == lowBuckets[SW].end()) continue;
                if (lowBuckets[NE].find(candidateSE_low ^ zLo) == lowBuckets[NE].end()) continue;
                if (lowBuckets[NW].find(candidateSE_low ^ xzLo) == lowBuckets[NW].end()) continue;

                anyFoundLow = true;
                candidateCount.fetch_add(1, memory_order_relaxed);

                // Iterate full bases for this low value
                for (uint32_t baseSE : lowBuckets[SE][candidateSE_low]) {
                    if (!isValidBase(baseSE, xMul, zMul)) continue;
                    uint32_t baseO = seed ^ baseSE;
                    BestSolution solution = nearestSolution(seed, xMul, zMul, baseO);
                    if (solution.distanceSquared < localBest.distanceSquared) {
                        localBest = solution;
                    }
                    foundCount.fetch_add(1, memory_order_relaxed);
                    break;
                }
                if (localBest.distanceSquared < UINT64_MAX) break;
            }
            if (!anyFoundLow) {
                ++localProcessed;
                if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                    searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                    localProcessed = 0;
                }
                continue;
            }

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
             << " originChunk=(" << best.chunkX << "," << best.chunkZ << ")"
             << " distance^2=" << best.distanceSquared << "\n";
    } else {
        cerr << "No valid solution found in scanned range." << '\n';
    }
    int a;
    cin>>a;
    return 0;
}
