// Compile:
//g++ -Ofast -march=native -mtune=native -flto -DNDEBUG -pthread -o wellfinder wellfinder.cpp
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
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <functional>

using namespace std;
using namespace chrono;

// ---------- Global control flags --------------------------------------------
static atomic<bool> stopRequested{false};
static atomic<uint64_t> foundCount{0};
static atomic<uint64_t> candidateCount{0};
static mutex outputMutex;
static mutex fileMutex;

// ---------- Bedrock RNG (well.h) --------------------------------------------
namespace DW {

constexpr uint32_t imul32(uint32_t a, uint32_t b) noexcept {
    return uint32_t(int64_t(int32_t(a)) * int64_t(int32_t(b)));
}

struct MTRandom {
    uint32_t mt[624];
    int index = 624;
    explicit MTRandom(uint32_t seed) noexcept { seed32(seed); }
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
            if (y & 1u) m ^= 0x9908b0dfu;
            mt[i] = m;
        }
        index = 0;
    }
    uint32_t random_int() noexcept {
        if (index >= 624) twist();
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
    uint32_t next31() noexcept { return rng.random_int() >> 1; }
    template <uint32_t Bound>
    uint32_t next() noexcept { return rng.random_int() % Bound; }
    uint32_t next(uint32_t bound) noexcept { return rng.random_int() % bound; }
};

constexpr uint32_t strHash(const char *text) noexcept {
    int32_t value = -2078137563;
    while (*text) value = int32_t(imul32(uint32_t(value), 435u) ^ int32_t(static_cast<unsigned char>(*text++)));
    return uint32_t(value);
}

static constexpr uint32_t FEATURE_KEY = strHash("minecraft:desert_after_surface_desert_well_feature");

struct FeatureSeed { uint32_t seedLow32, xMul, zMul, fKey; };

inline FeatureSeed DWMakeFeatureSeed(int64_t worldSeed) noexcept {
    uint32_t seedLow32 = uint32_t(int32_t(worldSeed));
    RNG rng(seedLow32);
    return FeatureSeed{seedLow32, rng.next31() | 1u, rng.next31() | 1u, FEATURE_KEY};
}

inline uint32_t fHash(const FeatureSeed &seed, int32_t chunkX, int32_t chunkZ) noexcept {
    int32_t combined = int32_t(imul32(uint32_t(chunkX), seed.xMul))
                       + int32_t(imul32(uint32_t(chunkZ), seed.zMul));
    uint32_t base = uint32_t(uint32_t(seed.seedLow32) ^ uint32_t(combined));
    return uint32_t(base ^ (seed.fKey + (base << 6) + (base >> 2) - 1640531527u));
}

inline bool isWell(int64_t worldSeed, int chunkX, int chunkZ) noexcept {
    FeatureSeed seed = DWMakeFeatureSeed(worldSeed);
    uint32_t regionSeed = fHash(seed, chunkX, chunkZ);
    RNG rng(regionSeed);
    return rng.next<500>() == 0u;
}

} // namespace DW

// ---------- Configurable split parameters ------------------------------------
static uint32_t LOWER_BITS = 22;
static uint32_t LOWER_SIZE = 1u << LOWER_BITS;
static uint32_t LOWER_MASK = LOWER_SIZE - 1u;
static uint32_t MITM_HIGH_BITS = 12;
static uint32_t MITM_HIGH_SIZE = 1u << MITM_HIGH_BITS;
static uint32_t MITM_HIGH_MASK = MITM_HIGH_SIZE - 1u;
static uint32_t MITM_LOW_BITS = 10;
static uint32_t MITM_LOW_SIZE = 1u << MITM_LOW_BITS;
static uint32_t MITM_LOW_MASK = MITM_LOW_SIZE - 1u;

static constexpr uint64_t TOTAL_UINT32 = 0x1'0000'0000ULL;
static constexpr uint32_t REPORT_INTERVAL_SECONDS = 20;
static constexpr uint64_t FLUSH_INTERVAL = 32;

// ---------- Mask structures ---------------------------------------------------
struct alignas(64) RM1024 { uint64_t w[32]{}; };
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
    inline void bitAnd(const M1024& o) noexcept {
        v[0]=_mm256_and_si256(v[0],o.v[0]); v[1]=_mm256_and_si256(v[1],o.v[1]);
        v[2]=_mm256_and_si256(v[2],o.v[2]); v[3]=_mm256_and_si256(v[3],o.v[3]);
    }
    inline void bitOr(const M1024& o) noexcept {
        v[0]=_mm256_or_si256(v[0],o.v[0]); v[1]=_mm256_or_si256(v[1],o.v[1]);
        v[2]=_mm256_or_si256(v[2],o.v[2]); v[3]=_mm256_or_si256(v[3],o.v[3]);
    }
    inline bool any() const noexcept {
        __m256i o01 = _mm256_or_si256(v[0], v[1]);
        __m256i o23 = _mm256_or_si256(v[2], v[3]);
        return !_mm256_testz_si256(_mm256_or_si256(o01, o23), _mm256_or_si256(o01, o23));
    }
};

static vector<RM1024> cm0, cm1;
static vector<M1024> avxMask0, avxMask1;

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

    if (x & 1u) { auto mask = _mm256_set1_epi64x(0x5555555555555555ull); v0=swapBits256(v0,mask,1); v1=swapBits256(v1,mask,1); v2=swapBits256(v2,mask,1); v3=swapBits256(v3,mask,1); }
    if (x & 2u) { auto mask = _mm256_set1_epi64x(0x3333333333333333ull); v0=swapBits256(v0,mask,2); v1=swapBits256(v1,mask,2); v2=swapBits256(v2,mask,2); v3=swapBits256(v3,mask,2); }
    if (x & 4u) { auto mask = _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0full); v0=swapBits256(v0,mask,4); v1=swapBits256(v1,mask,4); v2=swapBits256(v2,mask,4); v3=swapBits256(v3,mask,4); }
    if (x & 8u) { auto mask = _mm256_set1_epi64x(0x00ff00ff00ff00ffull); v0=swapBits256(v0,mask,8); v1=swapBits256(v1,mask,8); v2=swapBits256(v2,mask,8); v3=swapBits256(v3,mask,8); }
    if (x & 16u){ auto mask = _mm256_set1_epi64x(0x0000ffff0000ffffull); v0=swapBits256(v0,mask,16);v1=swapBits256(v1,mask,16);v2=swapBits256(v2,mask,16);v3=swapBits256(v3,mask,16);}
    if (x & 32u){ auto mask = _mm256_set1_epi64x(0x00000000ffffffffull); v0=swapBits256(v0,mask,32);v1=swapBits256(v1,mask,32);v2=swapBits256(v2,mask,32);v3=swapBits256(v3,mask,32);}
    if (x & 64u) { v0=_mm256_permute4x64_epi64(v0,0xB1); v1=_mm256_permute4x64_epi64(v1,0xB1); v2=_mm256_permute4x64_epi64(v2,0xB1); v3=_mm256_permute4x64_epi64(v3,0xB1); }
    if (x & 128u){ v0=_mm256_permute4x64_epi64(v0,0x4E); v1=_mm256_permute4x64_epi64(v1,0x4E); v2=_mm256_permute4x64_epi64(v2,0x4E); v3=_mm256_permute4x64_epi64(v3,0x4E); }
    if (x & 256u){ swap(v0,v1); swap(v2,v3); }
    if (x & 512u){ swap(v0,v2); swap(v1,v3); }

    _mm256_storeu_si256((__m256i*)&m.w[0], v0);  _mm256_storeu_si256((__m256i*)&m.w[16], v0);
    _mm256_storeu_si256((__m256i*)&m.w[4], v1);  _mm256_storeu_si256((__m256i*)&m.w[20], v1);
    _mm256_storeu_si256((__m256i*)&m.w[8], v2);  _mm256_storeu_si256((__m256i*)&m.w[24], v2);
    _mm256_storeu_si256((__m256i*)&m.w[12], v3); _mm256_storeu_si256((__m256i*)&m.w[28], v3);
}

// ---------- Fast MT19937 helpers ------------------------------------------------
static constexpr uint32_t MT_A = 1812433253u, TWIST_B = 0x9908b0dfu;
inline uint32_t temper(uint32_t y) noexcept { y^=y>>11; y^=(y<<7)&2636928640u; y^=(y<<15)&4022730752u; y^=y>>18; return y; }
inline uint32_t twistOnce(uint32_t a,uint32_t b,uint32_t c) noexcept { uint32_t y=(a&0x80000000u)|(b&0x7fffffffu); uint32_t m=c^(y>>1); if(y&1u)m^=TWIST_B; return m; }
inline DW::FeatureSeed MakeFeatureSeed(uint32_t seedLow32) noexcept {
    uint32_t mt0=seedLow32, mt1=0,mt2=0,mt397=0,mt398=0, prev=seedLow32;
    for(uint32_t i=1;i<=398;++i){ prev=MT_A*(prev^(prev>>30))+i; if(i==1)mt1=prev; else if(i==2)mt2=prev; else if(i==397)mt397=prev; else if(i==398)mt398=prev; }
    uint32_t raw0=temper(twistOnce(mt0,mt1,mt397)), raw1=temper(twistOnce(mt1,mt2,mt398));
    return DW::FeatureSeed{seedLow32, (raw0>>1)|1u, (raw1>>1)|1u, DW::FEATURE_KEY};
}
inline uint32_t computeRegionSeedFromBase(uint32_t base) noexcept { return base ^ (DW::FEATURE_KEY + (base<<6) + (base>>2) - 1640531527u); }

// ---------- Nearest chunk solution ----------------------------------------------
struct BestSol { uint32_t seed,xMul,zMul,baseO; int32_t chunkX,chunkZ; uint64_t distance; };
static inline __int128 iabs128(__int128 v){ return v<0?-v:v; }
static inline __int128 floorDiv128(__int128 a,__int128 b){ __int128 q=a/b,r=a%b; if(r&&a<0)--q; return q; }
static inline __int128 ceilDiv128(__int128 a,__int128 b){ __int128 q=a/b,r=a%b; if(r&&a>0)++q; return q; }
static bool solveLinearDiophantine(int64_t a,int64_t b,int64_t c,int64_t& x0,int64_t& y0,int64_t& g) {
    auto extgcd=[&](auto self,int64_t aa,int64_t bb)->pair<int64_t,int64_t>{
        if(bb==0)return{1,0}; auto p=self(self,bb,aa%bb);
        __int128 x=p.second, y=(__int128)p.first-(__int128)(aa/bb)*p.second; return{(int64_t)x,(int64_t)y};
    };
    g=gcd(a,b); if(g<0)g=-g; if(g==0)return c==0; if(c%g!=0)return false;
    auto uv=extgcd(extgcd,a,b); __int128 scale=(__int128)c/g; x0=(int64_t)((__int128)uv.first*scale); y0=(int64_t)((__int128)uv.second*scale); return true;
}
static BestSol nearest(uint32_t seed,uint32_t xMul,uint32_t zMul,uint32_t combinedVal) {
    int64_t cx0=0,cz0=0,g=0;
    if(!solveLinearDiophantine(int64_t(xMul),int64_t(zMul),int64_t(int32_t(combinedVal)),cx0,cz0,g)) return BestSol{seed,xMul,zMul,combinedVal,0,0,UINT64_MAX};
    if(g<=0) return BestSol{seed,xMul,zMul,combinedVal,0,0,UINT64_MAX};
    int64_t stepX=int64_t(zMul)/g, stepZ=int64_t(xMul)/g;
    __int128 ax0=cx0,az0=cz0,sx=stepX,sz=stepZ;
    auto feasible=[&](uint64_t D,__int128& loK,__int128& hiK){
        __int128 d=(__int128)D;
        __int128 lo1=ceilDiv128(-d-ax0,sx), hi1=floorDiv128(d-ax0,sx);
        __int128 lo2=ceilDiv128(az0-d,sz), hi2=floorDiv128(az0+d,sz);
        loK=max(lo1,lo2); hiK=min(hi1,hi2); return loK<=hiK;
    };
    uint64_t upper=uint64_t(max(iabs128(ax0),iabs128(az0))), lo=0,hi=upper;
    while(lo<hi){ uint64_t mid=lo+((hi-lo)>>1); __int128 kLo,kHi; if(feasible(mid,kLo,kHi))hi=mid; else lo=mid+1; }
    __int128 kLo,kHi; if(!feasible(lo,kLo,kHi)) return BestSol{seed,xMul,zMul,combinedVal,0,0,UINT64_MAX};
    __int128 k=kLo; __int128 bestCx=ax0+k*sx, bestCz=az0-k*sz;
    return BestSol{seed,xMul,zMul,combinedVal, int32_t(bestCx),int32_t(bestCz), lo};
}

// ---------- Formation input -----------------------------------------------------
struct OffsetCriteria { vector<pair<int,int>> rangesX, rangesZ; };
static vector<pair<int,int>> chunkOffsets;
static vector<OffsetCriteria> criteria;
static vector<vector<uint32_t>> baseSets;   // index 0 = (0,0)

static void readFormation() {
    cout << "Enter number of chunk offsets: " << flush; int n; cin >> n;
    chunkOffsets.resize(n); criteria.resize(n); baseSets.resize(n);
    for(int i=0;i<n;++i){
        cout << "Offset " << i << " (dx dz): " << flush; cin >> chunkOffsets[i].first >> chunkOffsets[i].second;
        cout << "Number of valid X ranges for this chunk: " << flush; int rx; cin >> rx;
        criteria[i].rangesX.resize(rx);
        for(int r=0;r<rx;++r){ cout << "  X range " << r << " (min max): " << flush; cin >> criteria[i].rangesX[r].first >> criteria[i].rangesX[r].second; }
        cout << "Number of valid Z ranges: " << flush; int rz; cin >> rz;
        criteria[i].rangesZ.resize(rz);
        for(int r=0;r<rz;++r){ cout << "  Z range " << r << " (min max): " << flush; cin >> criteria[i].rangesZ[r].first >> criteria[i].rangesZ[r].second; }
    }
    int zeroIdx = -1;
    for(int i=0;i<n;++i) if(chunkOffsets[i].first==0 && chunkOffsets[i].second==0){ zeroIdx=i; break; }
    if(zeroIdx==-1){ cerr << "Error: No (0,0) offset provided.\n"; exit(1); }
    if(zeroIdx!=0){ swap(chunkOffsets[0],chunkOffsets[zeroIdx]); swap(criteria[0],criteria[zeroIdx]); }
    cout << "Formation recorded.\n";
}
static bool offsetInCriteria(int offX,int offZ,const OffsetCriteria& crit){
    bool xOk=false; for(auto& p:crit.rangesX) if(offX>=p.first && offX<=p.second){xOk=true;break;} if(!xOk)return false;
    for(auto& p:crit.rangesZ) if(offZ>=p.first && offZ<=p.second) return true; return false;
}

// ---------- Universal well base cache ------------------------------------------
static vector<uint32_t> wellBases;

static void generateWellBases(const string& filename) {
    uint64_t limit = TOTAL_UINT32;
    unsigned threadCount = thread::hardware_concurrency(); if(threadCount==0)threadCount=1;
    atomic<uint64_t> processed{0};
    cerr << "Scanning all region seeds for wells (this happens only once)...\n";

    auto worker = [&](unsigned tid){
        uint64_t block=(limit+threadCount-1)/threadCount, start=tid*block, end=min(start+block,limit);
        vector<uint32_t> local;
        uint64_t proc=0;
        for(uint64_t v=start; v<end && !stopRequested; ++v){
            uint32_t base = uint32_t(v);
            uint32_t regionSeed = computeRegionSeedFromBase(base);
            DW::RNG rng(regionSeed);
            if(rng.next<500>() == 0u) local.push_back(base);
            ++proc;
            if((proc & (FLUSH_INTERVAL-1)) == 0){
                processed.fetch_add(proc, memory_order_relaxed);
                proc = 0;
            }
        }
        processed.fetch_add(proc, memory_order_relaxed);
        lock_guard<mutex> g(fileMutex);
        wellBases.insert(wellBases.end(), local.begin(), local.end());
    };

    vector<thread> threads;
    for(unsigned t=0;t<threadCount;++t) threads.emplace_back(worker,t);
    thread rep([&](){
        while(processed.load() < limit && !stopRequested){
            this_thread::sleep_for(seconds(5));
            double pct = 100.0 * processed.load() / limit;
            cerr << fixed << setprecision(2) << pct << "%\n";
        }
    });
    for(auto& t:threads) t.join();
    stopRequested = true;
    rep.join();

    sort(wellBases.begin(), wellBases.end());
    wellBases.erase(unique(wellBases.begin(), wellBases.end()), wellBases.end());

    ofstream ofs(filename, ios::binary);
    uint64_t count = wellBases.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(wellBases.data()), count * sizeof(uint32_t));
    cerr << "Saved " << count << " well bases to " << filename << "\n";
}

static bool loadWellBases(const string& filename) {
    ifstream ifs(filename, ios::binary);
    if(!ifs) return false;
    uint64_t count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    wellBases.resize(count);
    ifs.read(reinterpret_cast<char*>(wellBases.data()), count * sizeof(uint32_t));
    if(ifs.gcount() != streamsize(count * sizeof(uint32_t))) {
        wellBases.clear();
        return false;
    }
    cerr << "Loaded " << wellBases.size() << " well bases from " << filename << "\n";
    return true;
}

static void buildOffsetSetsFromCache() {
    for(auto& s : baseSets) s.clear();

    unsigned threadCount = thread::hardware_concurrency(); if(threadCount==0)threadCount=1;
    size_t total = wellBases.size();
    size_t block = (total + threadCount - 1) / threadCount;

    vector<vector<vector<uint32_t>>> threadLocalSets(threadCount,
        vector<vector<uint32_t>>(chunkOffsets.size()));

    vector<thread> threads;
    for(unsigned t=0; t<threadCount; ++t){
        size_t start = t * block;
        size_t end = min(start + block, total);
        threads.emplace_back([start,end,t,&threadLocalSets](){
            auto& localSets = threadLocalSets[t];
            for(size_t i = start; i < end; ++i){
                uint32_t base = wellBases[i];
                uint32_t regionSeed = computeRegionSeedFromBase(base);
                DW::RNG rng(regionSeed);
                rng.next<500>(); // we know it's 0, just advance
                int offZ = rng.next<16>();
                int offX = rng.next<16>();
                for(size_t o = 0; o < criteria.size(); ++o){
                    if(offsetInCriteria(offX, offZ, criteria[o])){
                        localSets[o].push_back(base);
                    }
                }
            }
        });
    }
    for(auto& t: threads) t.join();

    for(size_t o = 0; o < criteria.size(); ++o){
        for(unsigned t = 0; t < threadCount; ++t){
            auto& part = threadLocalSets[t][o];
            baseSets[o].insert(baseSets[o].end(), part.begin(), part.end());
        }
        sort(baseSets[o].begin(), baseSets[o].end());
        baseSets[o].erase(unique(baseSets[o].begin(), baseSets[o].end()), baseSets[o].end());
    }
    cerr << "Base sets built from cache. Set sizes:";
    for(auto& s : baseSets) cerr << ' ' << s.size();
    cerr << '\n';
}

// ---------- Top‑10 solution tracking & file output -----------------------------
static const int TOP_K = 10;
static vector<BestSol> topSolutions;
static mutex topMutex;

static void addSolution(const BestSol& sol) {
    lock_guard<mutex> g(topMutex);
    for (const auto& t : topSolutions)
        if (t.seed==sol.seed && t.xMul==sol.xMul && t.zMul==sol.zMul && t.baseO==sol.baseO) return;
    auto it = lower_bound(topSolutions.begin(), topSolutions.end(), sol,
        [](const BestSol& a, const BestSol& b) { return a.distance < b.distance; });
    topSolutions.insert(it, sol);
    if (topSolutions.size() > TOP_K) topSolutions.pop_back();
}

static void saveTopSolutions(const string& filename) {
    lock_guard<mutex> g(topMutex);
    ofstream ofs(filename);
    if (!ofs) return;
    ofs << "Top " << topSolutions.size() << " solutions:\n";
    for (size_t i=0; i<topSolutions.size(); ++i) {
        const auto& s = topSolutions[i];
        ofs << "#" << i+1 << " seed=" << s.seed << " xMul=" << s.xMul << " zMul=" << s.zMul
            << " originChunk=(" << s.chunkX << ',' << s.chunkZ << ") distance=" << s.distance << '\n';
    }
}

static string getBestDisplayString() {
    lock_guard<mutex> g(topMutex);
    if (topSolutions.empty()) return "none";
    const auto& s = topSolutions[0];
    stringstream ss;
    ss << "best seed=" << s.seed << " dist=" << s.distance
       << " chunk=(" << s.chunkX << ',' << s.chunkZ << ')';
    return ss.str();
}

// ---------- Progress reporting --------------------------------------------------
struct Phase { string name; uint64_t total; atomic<uint64_t> processed{}; };
static void printProgress(const string& phaseName, uint64_t processed, uint64_t total, double rate) {
    lock_guard<mutex> g(outputMutex);
    double pct = total ? 100.0*processed/total : 0.0;
    cerr << '[' << phaseName << "] " << fixed << setprecision(3) << pct << "% (" << processed << '/' << total
         << ") rate=" << rate << "/s";
    if (phaseName=="search") {
        cerr << " found=" << foundCount << " candidates=" << candidateCount
             << "  " << getBestDisplayString();
    }
    cerr << '\n';
}
static void reporter(const Phase& status) {
    uint64_t lastProc=0; auto lastTime=steady_clock::now();
    while(!stopRequested && status.processed<status.total) {
        this_thread::sleep_for(seconds(REPORT_INTERVAL_SECONDS));
        if(stopRequested) break;
        auto now=steady_clock::now(); double elapsed=duration<double>(now-lastTime).count();
        uint64_t cur=status.processed; double rate = elapsed>0 ? (cur-lastProc)/elapsed : 0;
        printProgress(status.name, cur, status.total, rate);
        lastTime=now; lastProc=cur;
    }
    printProgress(status.name, status.processed, status.total, 0);
}

// ---------- Row tables & anchor bucket (parameterised) --------------------------
static vector<vector<RM1024>> baseRows;
static vector<uint32_t> anchorValuesFlat;
static vector<uint32_t> anchorBucketStart;
static vector<uint32_t> anchorBucketCount;

static void initMasks() {
    cm0.resize(MITM_LOW_SIZE);
    cm1.resize(MITM_LOW_SIZE);
    avxMask0.resize(MITM_LOW_SIZE);
    avxMask1.resize(MITM_LOW_SIZE);
    for (uint32_t d = 0; d < MITM_LOW_SIZE; ++d) {
        cm0[d] = RM1024{};
        uint32_t bits = MITM_LOW_SIZE - d;
        uint32_t fullWords = bits / 64, rem = bits % 64;
        for (uint32_t i = 0; i < fullWords; ++i) cm0[d].w[i] = ~0ull;
        if (rem && fullWords < 16) cm0[d].w[fullWords] = (1ull << rem) - 1ull;
        for (int i = 0; i < 16; ++i) cm1[d].w[i] = ~cm0[d].w[i];

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

static void buildRowsAndBucket(){
    baseRows.resize(baseSets.size());
    for(size_t i=0;i<baseSets.size();++i){
        baseRows[i].assign(MITM_HIGH_SIZE, RM1024{});
        for(uint32_t v:baseSets[i]){
            uint32_t low=v & LOWER_MASK, row=low>>MITM_LOW_BITS, col=low & MITM_LOW_MASK;
            baseRows[i][row].w[col>>6] |= 1ULL<<(col&63);
        }
    }
    const auto& set0 = baseSets[0];
    anchorValuesFlat = set0;
    sort(anchorValuesFlat.begin(), anchorValuesFlat.end(), [](uint32_t a,uint32_t b){
        uint32_t la=a&LOWER_MASK, lb=b&LOWER_MASK; if(la!=lb)return la<lb; return a<b;
    });
    anchorBucketStart.assign(LOWER_SIZE, 0);
    anchorBucketCount.assign(LOWER_SIZE, 0);
    for(size_t i=0;i<anchorValuesFlat.size();){
        uint32_t low = anchorValuesFlat[i]&LOWER_MASK; size_t j=i;
        while(j<anchorValuesFlat.size() && (anchorValuesFlat[j]&LOWER_MASK)==low) ++j;
        anchorBucketStart[low] = uint32_t(i);
        anchorBucketCount[low] = uint32_t(j-i);
        i=j;
    }
}

// ---------- Interactive tuning -----------------------------------------------
static void interactiveTune() {
    cout << "Interactive tuning: enter HIGH_BITS LOW_BITS [TIME_LIMIT] to benchmark.\n";
    cout << "Constraints: HIGH_BITS >= 1, 1 <= LOW_BITS <= 10, total (HIGH+LOW) <= 26.\n";
    cout << "  (e.g. 18 6 2  → test 18 high + 6 low for 2 seconds)\n";
    cout << "  Enter just HIGH_BITS LOW_BITS to finalise and start search.\n";
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<uint32_t> seedDist(0, UINT32_MAX);

    while (true) {
        cout << "> " << flush;
        string line; getline(cin, line);
        if (line.empty()) continue;
        istringstream iss(line);
        uint32_t hb, lb; double tlim = -1.0;
        if (iss >> hb >> lb) {
            if (hb < 1 || lb < 1 || lb > 10) {
                cerr << "Invalid: LOW_BITS must be between 1 and 10.\n";
                continue;
            }
            uint32_t totalBits = hb + lb;
            if (totalBits > 26) {
                cerr << "Invalid: total bits (high+low) must be ≤ 26 to keep memory usage manageable.\n";
                continue;
            }
            if (iss >> tlim) {
                if (tlim <= 0.0) { cerr << "Time limit must be positive.\n"; continue; }
                LOWER_BITS = totalBits;
                MITM_HIGH_BITS = hb; MITM_LOW_BITS = lb;
                MITM_HIGH_SIZE = 1 << hb; MITM_HIGH_MASK = MITM_HIGH_SIZE - 1;
                MITM_LOW_SIZE = 1 << lb; MITM_LOW_MASK = MITM_LOW_SIZE - 1;
                LOWER_SIZE = 1 << LOWER_BITS; LOWER_MASK = LOWER_SIZE - 1;
                initMasks();
                buildRowsAndBucket();

                atomic<bool> timeUp{false};
                thread timer([&](){ this_thread::sleep_for(duration<double>(tlim)); timeUp = true; });

                uint64_t seedsDone = 0;
                auto benchStart = steady_clock::now();
                while (!timeUp) {
                    uint32_t seed = seedDist(rng);
                    auto feat = MakeFeatureSeed(seed);
                    uint32_t xMul = feat.xMul, zMul = feat.zMul;
                    uint32_t seedLow = seed & LOWER_MASK;
                    uint32_t sl10 = seedLow & MITM_LOW_MASK;
                    uint32_t seedHigh12 = (seedLow >> MITM_LOW_BITS) & MITM_HIGH_MASK;

                    vector<vector<RM1024>> permRows(baseSets.size());
                    for (size_t o = 0; o < baseSets.size(); ++o) {
                        permRows[o].resize(MITM_HIGH_SIZE);
                        for (uint32_t r = 0; r < MITM_HIGH_SIZE; ++r) {
                            permRows[o][r] = baseRows[o][r];
                            xorPerm(permRows[o][r], sl10);
                        }
                    }

                    for (uint32_t row0 = 0; row0 < MITM_HIGH_SIZE && !timeUp; ++row0) {
                        M1024 mask;
                        mask.v[0] = _mm256_loadu_si256((__m256i*)&permRows[0][row0].w[0]);
                        mask.v[1] = _mm256_loadu_si256((__m256i*)&permRows[0][row0].w[4]);
                        mask.v[2] = _mm256_loadu_si256((__m256i*)&permRows[0][row0].w[8]);
                        mask.v[3] = _mm256_loadu_si256((__m256i*)&permRows[0][row0].w[12]);
                        bool skip = false;
                        for (size_t o = 1; o < baseSets.size(); ++o) {
                            int64_t D = int64_t(chunkOffsets[o].first)*xMul + int64_t(chunkOffsets[o].second)*zMul;
                            uint32_t Dlow = uint32_t(D) & LOWER_MASK;
                            uint32_t dLow10 = Dlow & MITM_LOW_MASK;
                            uint32_t dHigh12 = (Dlow >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                            uint32_t uHigh12 = seedHigh12 ^ row0;
                            uint32_t sumHigh = (uHigh12 + dHigh12) & MITM_HIGH_MASK;
                            uint32_t targetRow = seedHigh12 ^ sumHigh;

                            auto getShift = [](unsigned r, uint32_t lowMask)->tuple<unsigned,unsigned,unsigned>{
                                r &= lowMask; return { r>>6, r&63u, 64u-(r&63u) };
                            };
                            auto [wordShift, bitShift, bitShift2] = getShift(dLow10, MITM_LOW_MASK);
                            bool doShift = (bitShift != 0);
                            __m128i shiftR = _mm_cvtsi32_si128(bitShift), shiftL = _mm_cvtsi32_si128(bitShift2);
                            M1024 extra;
                            extra.loadRot(permRows[o][targetRow].w, wordShift, shiftR, shiftL, doShift);
                            extra.bitAnd(avxMask0[dLow10]);
                            if (dLow10 != 0u) {
                                uint32_t targetRow1 = seedHigh12 ^ ((sumHigh+1u)&MITM_HIGH_MASK);
                                M1024 extra1;
                                extra1.loadRot(permRows[o][targetRow1].w, wordShift, shiftR, shiftL, doShift);
                                extra1.bitAnd(avxMask1[dLow10]);
                                extra.bitOr(extra1);
                            }
                            mask.bitAnd(extra);
                            if (!mask.any()) { skip = true; break; }
                        }
                        if (skip) continue;

                        alignas(32) uint64_t mw[16];
                        _mm256_storeu_si256((__m256i*)&mw[0], mask.v[0]);
                        _mm256_storeu_si256((__m256i*)&mw[4], mask.v[1]);
                        _mm256_storeu_si256((__m256i*)&mw[8], mask.v[2]);
                        _mm256_storeu_si256((__m256i*)&mw[12], mask.v[3]);
                        for (int w = 0; w < 16 && !timeUp; ++w) {
                            uint64_t bits = mw[w];
                            while (bits && !timeUp) {
                                unsigned bit = unsigned(__builtin_ctzll(bits)); bits &= bits-1;
                                uint32_t uLow10 = uint32_t(w*64 + bit);
                                uint32_t uHigh12 = seedHigh12 ^ row0;
                                uint32_t lowVal = seedLow ^ ((uHigh12<<MITM_LOW_BITS) | uLow10);

                                uint32_t startIdx = anchorBucketStart[lowVal];
                                uint32_t cnt = anchorBucketCount[lowVal];
                                for (uint32_t k = 0; k < cnt && !timeUp; ++k) {
                                    uint32_t base0 = anchorValuesFlat[startIdx + k];
                                    uint32_t U = seed ^ base0;
                                    bool ok = true;
                                    for (size_t o = 1; o < baseSets.size(); ++o) {
                                        int64_t Dv = int64_t(chunkOffsets[o].first)*xMul + int64_t(chunkOffsets[o].second)*zMul;
                                        uint32_t targetBase = seed ^ uint32_t(U + uint32_t(Dv));
                                        if (!binary_search(baseSets[o].begin(), baseSets[o].end(), targetBase)) { ok = false; break; }
                                    }
                                }
                            }
                        }
                    }
                    ++seedsDone;
                }
                timer.join();
                auto benchEnd = steady_clock::now();
                double elapsed = duration<double>(benchEnd - benchStart).count();
                if (seedsDone == 0) cout << "No seeds processed.\n";
                else cout << seedsDone << " seeds in " << elapsed << " s (" << (elapsed*1e6/seedsDone) << " µs/seed)\n";
            } else {
                LOWER_BITS = totalBits;
                MITM_HIGH_BITS = hb; MITM_LOW_BITS = lb;
                MITM_HIGH_SIZE = 1 << hb; MITM_HIGH_MASK = MITM_HIGH_SIZE - 1;
                MITM_LOW_SIZE = 1 << lb; MITM_LOW_MASK = MITM_LOW_SIZE - 1;
                LOWER_SIZE = 1 << LOWER_BITS; LOWER_MASK = LOWER_SIZE - 1;
                cout << "Using HIGH=" << hb << " LOW=" << lb << " (total " << LOWER_BITS << " bits)\n";
                break;
            }
        } else cerr << "Invalid input.\n";
    }
}

// ---------- Search phase -------------------------------------------------------
static void searchWorker(uint64_t start, uint64_t end, bool testMode,
                         steady_clock::time_point searchStart, Phase& searchStatus) {
    uint64_t localProc=0;
    vector<vector<RM1024>> localPermRows(baseSets.size());
    for(size_t o=0;o<baseSets.size();++o) localPermRows[o].resize(MITM_HIGH_SIZE);
    const auto& refRows = baseRows;

    for(uint32_t sl10=0; sl10<MITM_LOW_SIZE; ++sl10){
        if(stopRequested) break;
        if(testMode && duration<double>(steady_clock::now()-searchStart).count()>=10.0){ stopRequested=true; break; }
        for(size_t o=0;o<baseSets.size();++o)
            for(uint32_t r=0;r<MITM_HIGH_SIZE;++r){
                localPermRows[o][r] = refRows[o][r];
                xorPerm(localPermRows[o][r], sl10);
            }

        for(uint64_t base=start; base<end; base+=MITM_LOW_SIZE){
            if(stopRequested) break;
            uint64_t curSeedVal = base+sl10;
            if(curSeedVal >= end) continue;
            uint32_t seed = uint32_t(curSeedVal);
            auto feat = MakeFeatureSeed(seed);
            uint32_t xMul=feat.xMul, zMul=feat.zMul;
            uint32_t seedLow = seed & LOWER_MASK;
            uint32_t seedHigh12 = (seedLow >> MITM_LOW_BITS) & MITM_HIGH_MASK;

            for(uint32_t row0=0; row0<MITM_HIGH_SIZE; ++row0){
                M1024 mask;
                mask.v[0]=_mm256_loadu_si256((__m256i*)&localPermRows[0][row0].w[0]);
                mask.v[1]=_mm256_loadu_si256((__m256i*)&localPermRows[0][row0].w[4]);
                mask.v[2]=_mm256_loadu_si256((__m256i*)&localPermRows[0][row0].w[8]);
                mask.v[3]=_mm256_loadu_si256((__m256i*)&localPermRows[0][row0].w[12]);
                bool skip=false;

                for(size_t o=1; o<baseSets.size(); ++o){
                    int64_t D = int64_t(chunkOffsets[o].first)*xMul + int64_t(chunkOffsets[o].second)*zMul;
                    uint32_t Dlow = uint32_t(D) & LOWER_MASK;
                    uint32_t dLow10 = Dlow & MITM_LOW_MASK;
                    uint32_t dHigh12 = (Dlow >> MITM_LOW_BITS) & MITM_HIGH_MASK;
                    uint32_t uHigh12 = seedHigh12 ^ row0;
                    uint32_t sumHigh = (uHigh12 + dHigh12) & MITM_HIGH_MASK;
                    uint32_t targetRow = seedHigh12 ^ sumHigh;

                    auto getShift = [](unsigned r, uint32_t lowMask)->tuple<unsigned,unsigned,unsigned>{
                        r &= lowMask; return { r>>6, r&63u, 64u-(r&63u) };
                    };
                    auto [wordShift, bitShift, bitShift2] = getShift(dLow10, MITM_LOW_MASK);
                    bool doShift = (bitShift != 0);
                    __m128i shiftR = _mm_cvtsi32_si128(bitShift), shiftL = _mm_cvtsi32_si128(bitShift2);

                    M1024 extra;
                    extra.loadRot(localPermRows[o][targetRow].w, wordShift, shiftR, shiftL, doShift);
                    extra.bitAnd(avxMask0[dLow10]);
                    if (dLow10 != 0u) {
                        uint32_t targetRow1 = seedHigh12 ^ ((sumHigh+1u)&MITM_HIGH_MASK);
                        M1024 extra1;
                        extra1.loadRot(localPermRows[o][targetRow1].w, wordShift, shiftR, shiftL, doShift);
                        extra1.bitAnd(avxMask1[dLow10]);
                        extra.bitOr(extra1);
                    }
                    mask.bitAnd(extra);
                    if (!mask.any()) { skip=true; break; }
                }
                if(skip) continue;

                alignas(32) uint64_t mw[16];
                _mm256_storeu_si256((__m256i*)&mw[0],mask.v[0]);
                _mm256_storeu_si256((__m256i*)&mw[4],mask.v[1]);
                _mm256_storeu_si256((__m256i*)&mw[8],mask.v[2]);
                _mm256_storeu_si256((__m256i*)&mw[12],mask.v[3]);

                for(int w=0; w<16; ++w){
                    uint64_t bits = mw[w];
                    while(bits){
                        unsigned bit = unsigned(__builtin_ctzll(bits)); bits &= bits-1;
                        uint32_t uLow10 = uint32_t(w*64 + bit);
                        uint32_t uHigh12 = seedHigh12 ^ row0;
                        uint32_t lowVal = seedLow ^ ((uHigh12<<MITM_LOW_BITS) | uLow10);

                        uint32_t startIdx = anchorBucketStart[lowVal];
                        uint32_t cnt = anchorBucketCount[lowVal];
                        for(uint32_t k=0; k<cnt; ++k){
                            uint32_t base0 = anchorValuesFlat[startIdx + k];
                            uint32_t U = seed ^ base0;
                            bool ok=true;
                            for(size_t o=1; o<baseSets.size(); ++o){
                                int64_t Dv = int64_t(chunkOffsets[o].first)*xMul + int64_t(chunkOffsets[o].second)*zMul;
                                uint32_t targetBase = seed ^ uint32_t(U + uint32_t(Dv));
                                if(!binary_search(baseSets[o].begin(), baseSets[o].end(), targetBase)){ ok=false; break; }
                            }
                            if(!ok) continue;

                            BestSol sol = nearest(seed, xMul, zMul, U);
                            if(sol.distance == UINT64_MAX) continue;

                            if(!DW::isWell(int64_t(seed), sol.chunkX, sol.chunkZ)) continue;
                            bool allWells=true;
                            for(size_t o=1; o<chunkOffsets.size(); ++o){
                                int testX = sol.chunkX + chunkOffsets[o].first;
                                int testZ = sol.chunkZ + chunkOffsets[o].second;
                                if(!DW::isWell(int64_t(seed), testX, testZ)){ allWells=false; break; }
                            }
                            if(!allWells) continue;

                            candidateCount.fetch_add(1); foundCount.fetch_add(1);
                            addSolution(sol);
                        }
                    }
                }
            }
            ++localProc;
            if((localProc & (FLUSH_INTERVAL-1)) == 0){
                searchStatus.processed += localProc;
                localProc = 0;
            }
        }
    }
    searchStatus.processed += localProc;
}

// ---------- Main (with fixed‑block resume) ------------------------------------
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    signal(SIGINT, [](int){ stopRequested = true; });

    readFormation();

    cout << "Enter mode (1 = normal, 2 = test): " << flush;
    int mode; cin >> mode;
    bool testMode = (mode == 2);

    // --- Universal well base cache ---
    string wellCacheFile = "well_bases.bin";
    if (!loadWellBases(wellCacheFile)) {
        if (!testMode) {
            cout << "No universal well cache found. Generate one now? (y/n): " << flush;
            char ans; cin >> ans;
            if (ans == 'y' || ans == 'Y') {
                generateWellBases(wellCacheFile);
            } else {
                cerr << "Cannot proceed without base data.\n";
                return 1;
            }
        } else {
            cerr << "Test mode requires the well cache; please generate it first in normal mode.\n";
            return 1;
        }
    }
    buildOffsetSetsFromCache();

    if (testMode && stopRequested) cerr << "Base scan time limit reached.\n";
    stopRequested.store(false);

    if (!testMode) {
        interactiveTune();
    } else {
        LOWER_BITS = 22; MITM_HIGH_BITS = 12; MITM_LOW_BITS = 10;
        MITM_HIGH_SIZE = 1 << 12; MITM_HIGH_MASK = MITM_HIGH_SIZE - 1;
        MITM_LOW_SIZE = 1024; MITM_LOW_MASK = 1023;
        LOWER_SIZE = 1 << 22; LOWER_MASK = LOWER_SIZE - 1;
    }
    initMasks();
    buildRowsAndBucket();

    // --- Fixed‑block decomposition for the full 2^32 space ---
    uint64_t totalSeeds = TOTAL_UINT32;
    unsigned threadCount = thread::hardware_concurrency(); if(threadCount==0) threadCount=1;
    // Full‑scan block size (used for both fresh and resume)
    uint64_t fullBlockSize = (totalSeeds + threadCount - 1) / threadCount;
    fullBlockSize = (fullBlockSize + MITM_LOW_MASK) & ~uint64_t(MITM_LOW_MASK);

    uint64_t skipSeeds = 0;
    if (!testMode) {
        cout << "Resume from a previous percentage? (y/n): " << flush;
        char ans; cin >> ans;
        if (ans == 'y' || ans == 'Y') {
            double pct;
            cout << "Enter percentage already scanned (e.g. 3.1): " << flush;
            cin >> pct;
            uint64_t rawSkip = uint64_t(pct * 0.01 * totalSeeds);
            // Align to MITM_LOW_SIZE to avoid mid‑seed issues
            skipSeeds = (rawSkip / MITM_LOW_SIZE) * MITM_LOW_SIZE;
            cerr << "Resuming from seed " << skipSeeds << " (aligned).\n";
        }
    }

    // Compute block index containing skipSeeds
    uint64_t blockIndex = skipSeeds / fullBlockSize;   // block that was partially or fully done
    uint64_t offsetInBlock = skipSeeds % fullBlockSize; // where to start inside that block

    // Build list of remaining block indices (blockIndex .. threadCount-1)
    vector<unsigned> blockIndices;
    for (unsigned b = blockIndex; b < threadCount; ++b) {
        blockIndices.push_back(b);
    }
    // Shuffle the block indices (except keep the first one at front if we need to start mid‑block)
    // We'll handle the first block specially if offsetInBlock > 0.
    // Shuffle all indices; then manually ensure the partial block is processed correctly.
    {
        random_device rd;
        mt19937 rng(rd());
        shuffle(blockIndices.begin(), blockIndices.end(), rng);
    }

    // Prepare thread ranges
    vector<pair<uint64_t,uint64_t>> threadRanges(blockIndices.size());
    for (size_t i = 0; i < blockIndices.size(); ++i) {
        unsigned b = blockIndices[i];
        uint64_t blockStart = uint64_t(b) * fullBlockSize;
        uint64_t blockEnd = min(blockStart + fullBlockSize, totalSeeds);
        if (b == blockIndex && offsetInBlock > 0) {
            // This is the partial block where we resume
            blockStart = skipSeeds;   // start from skipSeeds
        }
        threadRanges[i] = {blockStart, blockEnd};
    }

    uint64_t remaining = totalSeeds - skipSeeds;
    Phase searchStatus{"search", remaining};
    vector<thread> searchThreads;
    auto searchStart = steady_clock::now();

    for (size_t i = 0; i < threadRanges.size(); ++i) {
        uint64_t start = threadRanges[i].first;
        uint64_t end = threadRanges[i].second;
        searchThreads.emplace_back([&, start, end]() {
            searchWorker(start, end, testMode, searchStart, searchStatus);
        });
    }

    atomic<bool> fileSaverStop{false};
    thread fileSaver([&]() {
        while (!fileSaverStop && !stopRequested) {
            this_thread::sleep_for(seconds(60));
            if (!stopRequested) saveTopSolutions("top_seeds.txt");
        }
    });

    thread rep(reporter, cref(searchStatus));
    for (auto& t : searchThreads) t.join();
    stopRequested = true;
    fileSaverStop = true;
    if (rep.joinable()) rep.join();
    if (fileSaver.joinable()) fileSaver.join();

    saveTopSolutions("top_seeds.txt");

    if (!topSolutions.empty()) {
        cerr << "Final top " << topSolutions.size() << " solutions:\n";
        for (size_t i = 0; i < topSolutions.size(); ++i) {
            const auto& sol = topSolutions[i];
            cerr << " #" << i+1 << " seed=" << sol.seed << " xMul=" << sol.xMul << " zMul=" << sol.zMul
                 << " originChunk=(" << sol.chunkX << ',' << sol.chunkZ << ") distance=" << sol.distance << '\n';
        }
    } else cerr << "No valid solution found.\n";

    cout << "Press Enter to exit." << flush; cin.ignore(); cin.get();
    return 0;
}
