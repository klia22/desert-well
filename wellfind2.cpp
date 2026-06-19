#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

/*
Seedfinding with 3 phases:
Phase 1: Base scan - precompute all base seeds and find valid ones to store.
Phase 2: Build table - precompute all values for odd multipliers via a N^2/4 scan
Phase 3: Query Multipliers - for every seed, extract its multiplier and query the table
*/

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


// Class type for storing and finding APs
class findAP {
public:
    using u32 = uint32_t;
    using u64 = uint64_t;

    static constexpr u32 TARGET_LENGTH = 6; // change to 5 later if needed

    findAP() = default;

    void insert(u32 element) {
        if (built_) {
            throw runtime_error("insert() called after build/query phase started");
        }
        values_.push_back(element);
    }

    void finalize() {
        std::call_once(build_once_, [this]() { build(); });
    }

    // Empty vector means no solution.
    vector<u32> query(u32 X) {
        finalize();

        if (X == 0 || (X & 1u) == 0u) return {};
        if (valid_.empty()) return {};

        auto it = lower_bound(
            valid_.begin(), valid_.end(), X,
            [](const pair<u32, vector<u32>>& a, u32 x) {
                return a.first < x;
            }
        );

        if (it == valid_.end() || it->first != X) return {};
        return it->second;
    }

private:
    vector<u32> values_;
    vector<u32> odds_;
    vector<u32> evens_;

    // Full membership bitset for 2^32 values: 2^32 bits / 64 = 2^26 words.
    vector<u64> present_words_;

    // One bit per present 64-bit word, to avoid unnecessary main-bitset loads.
    // There are 2^26 words, so summary has 2^26 bits / 64 = 2^20 words.
    vector<u64> live_words_;

    // Valid X -> full AP sequence.
    vector<pair<u32, vector<u32>>> valid_;
    unordered_set<u32> seen_x_;

    mutable std::once_flag build_once_;
    bool built_ = false;

    static inline bool bit_test(const u64* words, u32 idx) {
        return (words[idx >> 6] >> (idx & 63u)) & 1ULL;
    }
struct alignas(64) ThreadState {
        vector<pair<u32, vector<u32>>> valid;
        unordered_set<u32> seen_x;
        std::atomic<u64> processed{0};
    };

    void build() {
        if (built_) return;

        cerr << "Building AP lookup table...\n";

        sort(values_.begin(), values_.end());
        values_.erase(unique(values_.begin(), values_.end()), values_.end());

        const size_t n = values_.size();

        odds_.clear();
        evens_.clear();
        odds_.reserve(n / 2 + 1);
        evens_.reserve(n / 2 + 1);

        present_words_.assign(1ull << 26, 0ULL);
        live_words_.assign(1ull << 20, 0ULL);

        auto set_present = [&](u32 x) {
            const u32 word = x >> 6;
            const u32 bit  = x & 63u;
            present_words_[word] |= (1ULL << bit);

            const u32 live_word = word >> 6;
            const u32 live_bit  = word & 63u;
            live_words_[live_word] |= (1ULL << live_bit);
        };

        for (u32 x : values_) {
            set_present(x);
            if (x & 1u) odds_.push_back(x);
            else        evens_.push_back(x);
        }

        const u64* present = present_words_.data();
        const u64* live    = live_words_.data();

        auto present_fast = [&](u32 x) -> bool {
            const u32 word = x >> 6;
            const u32 live_word = word >> 6;
            const u64 live_mask = 1ULL << (word & 63u);

            if ((live[live_word] & live_mask) == 0ULL) return false;
            return bit_test(present, x);
        };

        auto capture_full_chain = [&](u32 seed, u32 step) -> vector<u32> {
            vector<u32> seq;
            seq.reserve(16);

            u32 start = seed;
            for (;;) {
                u32 prev = start - step; 
                if (!present_fast(prev)) break;
                start = prev;
            }

            u32 cur = start;
            for (;;) {
                if (!present_fast(cur)) break;
                seq.push_back(cur);

                u32 nxt = cur + step; 
                if (nxt == start) break; 
                cur = nxt;
            }
            return seq;
        };

        const u32* odd_ptr = odds_.data();
        const u32* even_ptr = evens_.data();
        const size_t odd_n = odds_.size();
        const size_t even_n = evens_.size();

        // -------------------------------------------------------------
        // Multithreading Setup
        // -------------------------------------------------------------
        const unsigned int num_threads = std::thread::hardware_concurrency();
        vector<ThreadState> t_states(num_threads);
        std::atomic<size_t> next_odd_idx{0};
        std::atomic<bool> workers_done{false};

        const auto t0 = chrono::steady_clock::now();

        // Worker lambda
        auto worker_task = [&](int tid) {
            auto& state = t_states[tid];
            const size_t chunk_size = 32; // Small chunks for optimal load balancing

            while (true) {
                size_t start_idx = next_odd_idx.fetch_add(chunk_size, std::memory_order_relaxed);
                if (start_idx >= odd_n) break;
                size_t end_idx = std::min(start_idx + chunk_size, odd_n);

                u64 local_processed_chunk = 0;

                for (size_t i = start_idx; i < end_idx; ++i) {
                    const u32 ai = odd_ptr[i];
                    
                    // Binary search replaces the monotonic linear scan, allowing dynamic chunks
                    auto it = std::upper_bound(even_ptr, even_ptr + even_n, ai);
                    size_t even_start = std::distance(even_ptr, it);

                    for (size_t j = even_start; j < even_n; ++j) {
                        const u32 aj = even_ptr[j];
                        const u32 X = aj - ai;

                        const u32 fwd = aj + X;
                        const u32 bwd = ai - X;

                        if (!present_fast(fwd) && !present_fast(bwd)) {
                            ++local_processed_chunk;
                            continue;
                        }

                        u32 count = 2;
                        u32 curr = fwd;
                        while (count < TARGET_LENGTH && present_fast(curr)) {
                            ++count;
                            curr += X;
                        }

                        curr = bwd;
                        while (count < TARGET_LENGTH && present_fast(curr)) {
                            ++count;
                            curr -= X;
                        }

                        if (count >= TARGET_LENGTH) {
                            if (state.seen_x.insert(X).second) {
                                vector<u32> seq = capture_full_chain(ai, X);
                                state.valid.emplace_back(X, std::move(seq));
                            }

                            u32 neg = 0u - X; 
                            if (state.seen_x.insert(neg).second) {
                                vector<u32> seq = capture_full_chain(ai, X);
                                reverse(seq.begin(), seq.end());
                                state.valid.emplace_back(neg, std::move(seq));
                            }
                        }
                        ++local_processed_chunk;
                    }
                    
                    // Update atomic state periodically to avoid hammering the cache line
                    state.processed.fetch_add(local_processed_chunk, std::memory_order_relaxed);
                    local_processed_chunk = 0;
                }
            }
        };

        // Reporter lambda
        auto reporter_task = [&]() {
            while (!workers_done.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                
                u64 total_done = 0;
                size_t total_valid = 0;
                for (int t = 0; t < num_threads; ++t) {
                    total_done += t_states[t].processed.load(std::memory_order_relaxed);
                    total_valid += t_states[t].valid.size();
                }

                const auto now = chrono::steady_clock::now();
                const double sec = chrono::duration<double>(now - t0).count();
                const double rate = sec > 0.0 ? double(total_done) / sec : 0.0;
                
                cerr << fixed << setprecision(2)
                     << "[pair-scan] Processed " << total_done << " pairs in " << sec << " s"
                     << " (" << (rate / 1e6) << " M pairs/s)"
                     << ", found " << total_valid << " valid X values\n";
            }
        };

        // Launch Threads
        vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker_task, i);
        }
        std::thread reporter(reporter_task);

        // Wait for workers
        for (auto& t : threads) t.join();
        
        workers_done.store(true, std::memory_order_relaxed);
        reporter.join();

        // -------------------------------------------------------------
        // Merge Phase
        // -------------------------------------------------------------
        seen_x_.clear();
        seen_x_.reserve(1024);
        valid_.clear();
        valid_.reserve(1024);

        for (int i = 0; i < num_threads; ++i) {
            for (auto& p : t_states[i].valid) {
                // Deduplicate globally in case two threads found different pairs 
                // resulting in the exact same X difference.
                if (seen_x_.insert(p.first).second) {
                    valid_.push_back(std::move(p));
                }
            }
        }

        sort(valid_.begin(), valid_.end(),
             [](const pair<u32, vector<u32>>& a, const pair<u32, vector<u32>>& b) {
                 return a.first < b.first;
             });

        values_.clear();
        values_.shrink_to_fit();
        odds_.shrink_to_fit();
        evens_.shrink_to_fit();
        seen_x_.clear();
        seen_x_.rehash(0);

        built_ = true;

        cerr << "Build complete. Valid X values: " << valid_.size() << "\n";
    }
};

uint64_t limitTotal = 1ull << 32;
static constexpr uint32_t REPORT_INTERVAL_SECONDS = 20;
static constexpr uint64_t FLUSH_INTERVAL = 20;
std::atomic<bool> stopRequested{false};
std::mutex outputMutex;

struct Phase {
    const char* name;
    uint64_t total;
    std::atomic<uint64_t> processed{0};
};

Phase baseStatus{"[base-scan]",limitTotal};
Phase searchStatus{"[search]",limitTotal};

    
inline void printProgress(const string &phaseName, uint64_t processed, uint64_t total, double rate) {
    lock_guard<mutex> guard(outputMutex);
    double percent = total ? (100.0 * processed / double(total)) : 0.0;
    cerr << '[' << phaseName << "] "
         << fixed << setprecision(3) << percent << "% (" << processed << '/' << total << ')'
         << " rate=" << rate << "/s";
}

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

//Linear Diplotomic solver for finding nearest AP in multiplier space

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

int main() {
    //Initialize APfind
    findAP apfinder;
    //Phase 1: base-scan
    cout << "Starting Phase 1: Base Scan\n";

    unsigned threadCount = thread::hardware_concurrency();
    if (threadCount == 0) threadCount = 1;

    auto baseWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = uint64_t(tid) * blockSize;
        uint64_t end = min(start + blockSize, limitTotal);

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
            
            apfinder.insert(base);

            ++localProcessed;
            if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                localProcessed = 0;
            }
        }

        baseStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
        lock_guard<mutex> guard(outputMutex);
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

    cerr << "Base Scan Complete\n";
    cerr << "Starting Phase 2: Building AP Table via pair scan\n";

    // Phase 2: Build AP table
    apfinder.finalize();

    cerr << "AP Table Build Complete. Valid X values: " << apfinder.query(1).size() << "\n";
    cerr << "Starting Phase 3: Querying Multipliers\n";

    auto searchWorker = [&](unsigned tid) {
        uint64_t blockSize = (limitTotal + threadCount - 1) / threadCount;
        uint64_t start = min(uint64_t(tid) * blockSize, limitTotal);
        uint64_t end = min(start + blockSize, limitTotal);

        uint64_t localProcessed = 0;
        BestSol localBest{0, 0, 0, 0, 0, 0, UINT64_MAX};

        for (uint64_t value = start; value < end && !stopRequested.load(memory_order_relaxed); ++value) {
            auto feat = MakeFeatureSeed(uint32_t(value));
            auto xMul = feat.xMul;
            auto zMul = feat.zMul;
            auto xresult = apfinder.query(xMul);
            auto zresult = apfinder.query(zMul);
            if(!xresult.empty()){
                for(auto baseO : xresult){
                    BestSol solution = nearest(feat.seedLow32, xMul, zMul, baseO);
                    if (solution.distance < localBest.distance) {
                        localBest = solution;
                    }
                }
            }
            if(!zresult.empty()){
                for(auto baseO : zresult){
                    BestSol solution = nearest(feat.seedLow32, xMul, zMul, baseO);
                    if (solution.distance < localBest.distance) {
                        localBest = solution;
                    }
                }
            }
            ++localProcessed;
            if ((localProcessed & (FLUSH_INTERVAL - 1)) == 0) {
                searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
                localProcessed = 0;
            }
            searchStatus.processed.fetch_add(localProcessed, memory_order_relaxed);
        }
    };

    vector<thread> searchThreads;
    searchThreads.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t) {
        searchThreads.emplace_back(searchWorker, t);
    }
    thread searchReporter(reporter, cref(searchStatus));

    for (auto &t : searchThreads) t.join();
    stopRequested.store(false, memory_order_relaxed);
    if (searchReporter.joinable()) searchReporter.join();

    cout << "Search Complete\n";
    if (localBest.distance < UINT64_MAX) {
        cerr << "FOUND seed=" << localBest.seed
                << " xMul=" << localBest.xMul
                << " zMul=" << localBest.zMul
                << " originChunk=(" << localBest.chunkX << ',' << localBest.chunkZ <<
                ") distance=" << localBest.distance << '\n';
    } else {
        cerr << "No valid seed found within the search space.\n";
    }
}