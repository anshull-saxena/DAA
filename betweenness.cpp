/*
 * Betweenness Centrality -- Brandes (2001)
 * "A Faster Algorithm for Betweenness Centrality"
 * Journal of Mathematical Sociology, 25(2), pp. 163-177
 *
 * - Undirected graph, duplicates removed, self-loops ignored
 * - CB divided by 2 (undirected correction)
 * - OpenMP parallel, per-thread buffers, dynamic scheduling
 * - Exact for n <= 100000; approximate (2048 random sources) for larger
 *
 * Compile:
 *   g++ -O3 -march=native -std=c++17 -fopenmp betweenness.cpp -o betweenness
 * Usage:
 *   ./betweenness <edge-list-file>
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <omp.h>

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace std;

static constexpr int EXACT_THRESHOLD = 100000;
static constexpr int SAMPLE_SIZE     = 512;

struct PairHash {
    size_t operator()(const pair<int,int>& p) const {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) * 2654435761ULL);
    }
};

struct Graph {
    vector<vector<int>> adj;
    vector<int>         originalId;
    long long           m = 0;
};

static long long peakRSS_KB() {
#if defined(__linux__)
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
#else
    return 0;
#endif
}

static void showProgress(int done, int total,
                         const chrono::steady_clock::time_point& t0) {
    if (total <= 0) return;
    constexpr int W = 40;
    double pct     = 100.0 * done / total;
    int    filled  = max(0, min(W, (int)(pct / 100.0 * W)));
    double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    double eta     = done > 0 ? elapsed / done * (total - done) : 0.0;
    cerr << "\r[" << string(filled,'#') << string(W-filled,' ') << "] "
         << fixed << setprecision(1) << pct << "% "
         << "(" << done << "/" << total << ") "
         << "ETA " << setprecision(0) << eta << "s   " << flush;
    if (done >= total) cerr << "\n";
}

static Graph loadGraph(const string& filename) {
    ifstream fin(filename);
    if (!fin) throw runtime_error("Cannot open: " + filename);

    unordered_map<int,int> idMap;
    idMap.reserve(1 << 20);
    unordered_set<pair<int,int>, PairHash> edgeSet;
    edgeSet.reserve(1 << 22);

    Graph G;
    int nextId = 0;

    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int a, b;
        if (!(iss >> a >> b)) continue;
        if (a == b) continue;

        auto mapNode = [&](int orig) -> int {
            auto it = idMap.find(orig);
            if (it != idMap.end()) return it->second;
            int idx = nextId++;
            idMap[orig] = idx;
            G.adj.emplace_back();
            G.originalId.push_back(orig);
            return idx;
        };

        int u = mapNode(a), v = mapNode(b);
        if (u > v) swap(u, v);
        edgeSet.insert({u, v});
    }

    for (auto& [lo, hi] : edgeSet) {
        G.adj[lo].push_back(hi);
        G.adj[hi].push_back(lo);
    }
    G.m = (long long)edgeSet.size();
    return G;
}

static vector<double> brandesFromSources(const Graph& G,
                                          const vector<int>& sources) {
    const int n     = (int)G.adj.size();
    const int total = (int)sources.size();
    vector<double> cb(n, 0.0);

    const auto t0 = chrono::steady_clock::now();
    atomic<int> progress{0};
    const int   pinterval = max(1, total / 200);

    #pragma omp parallel
    {
        vector<double>      sigma(n, 0.0);
        vector<int>         dist(n, -1);
        vector<double>      delta(n, 0.0);
        vector<vector<int>> pred(n);
        vector<int>         stk, vis;
        vector<double>      lcb(n, 0.0);

        stk.reserve(n);
        vis.reserve(n);

        #pragma omp for schedule(dynamic, 16) nowait
        for (int si = 0; si < total; ++si) {
            int s = sources[si];
            queue<int> Q;
            stk.clear(); vis.clear();

            sigma[s] = 1.0; dist[s] = 0;
            vis.push_back(s); Q.push(s);

            while (!Q.empty()) {
                int v = Q.front(); Q.pop();
                stk.push_back(v);
                for (int w : G.adj[v]) {
                    if (dist[w] < 0) {
                        dist[w] = dist[v] + 1;
                        Q.push(w); vis.push_back(w);
                    }
                    if (dist[w] == dist[v] + 1) {
                        sigma[w] += sigma[v];
                        pred[w].push_back(v);
                    }
                }
            }

            for (auto it = stk.rbegin(); it != stk.rend(); ++it) {
                int w = *it;
                for (int v : pred[w])
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                if (w != s) lcb[w] += delta[w];
            }

            for (int v : vis) {
                sigma[v] = 0.0; dist[v] = -1; delta[v] = 0.0;
                pred[v].clear();
            }

            int done = ++progress;
            if (done % pinterval == 0 || done == total) {
                #pragma omp critical
                showProgress(done, total, t0);
            }
        }

        #pragma omp critical
        for (int i = 0; i < n; ++i) cb[i] += lcb[i];
    }
    return cb;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <edge-list-file>\n";
        return 1;
    }

    const string filename = argv[1];
    cerr << "Using " << omp_get_max_threads() << " OpenMP threads\n";

    // ---- Load ----
    auto tLoadStart = chrono::steady_clock::now();
    Graph G = loadGraph(filename);
    double loadTime = chrono::duration<double>(
        chrono::steady_clock::now() - tLoadStart).count();
    long long memAfterLoad = peakRSS_KB();

    int       n       = (int)G.adj.size();
    long long edges   = G.m;
    double    density = (n > 1)
        ? 100.0 * edges / ((long long)n * (n - 1) / 2)
        : 0.0;

    bool   is_approx = (n > EXACT_THRESHOLD);
    int    sample_sz = is_approx ? min(SAMPLE_SIZE, n) : n;
    string method    = is_approx ? "approximate" : "exact";

    // ---- Build source list ----
    vector<int> sources(sample_sz);
    if (!is_approx) {
        iota(sources.begin(), sources.end(), 0);
    } else {
        // Random sample without replacement, fixed seed for reproducibility
        mt19937 rng(42);
        vector<int> all(n);
        iota(all.begin(), all.end(), 0);
        shuffle(all.begin(), all.end(), rng);
        sources.assign(all.begin(), all.begin() + sample_sz);
    }

    // ---- Print graph info ----
    cout << "========================================\n";
    cout << "Dataset        : " << filename        << "\n";
    cout << "Graph type     : undirected\n";
    cout << "Method         : " << method          << "\n";
    cout << "Nodes          : " << n               << "\n";
    cout << "Edges          : " << edges           << "\n";
    cout << fixed << setprecision(6);
    cout << "Density        : " << density         << " %\n";
    cout << fixed << setprecision(3);
    cout << "Load time      : " << loadTime        << " s\n";
    if (is_approx)
        cout << "Samples        : " << sample_sz
             << " random sources (scores scaled by n/k)\n";
    cout << "========================================\n";

    // ---- Run Brandes ----
    cerr << "Running Brandes algorithm...\n";
    auto tBCStart = chrono::steady_clock::now();
    vector<double> cb = brandesFromSources(G, sources);
    double bcTime = chrono::duration<double>(
        chrono::steady_clock::now() - tBCStart).count();

    // Undirected correction -- each path counted twice
    for (double& v : cb) v /= 2.0;

    // Scale approximate scores by n/k
    if (is_approx) {
        double scale = (double)n / sample_sz;
        for (double& v : cb) v *= scale;
    }

    long long memTotal = peakRSS_KB();

    // ---- Sort ----
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    partial_sort(order.begin(), order.begin() + min(20, n), order.end(),
        [&](int a, int b){ return cb[a] > cb[b]; });

    // ---- Top 20 ----
    cout << "\n--- Top 20 Nodes by Betweenness Centrality ---\n";
    cout << left << setw(7)  << "Rank"
         << setw(14) << "Node ID"
         << "Betweenness Score\n";

    cout << fixed << setprecision(4);
    int limit = min(20, n);
    for (int i = 0; i < limit; ++i) {
        int idx = order[i];
        cout << left << setw(7)  << (i + 1)
             << setw(14) << G.originalId[idx]
             << cb[idx] << "\n";
    }

    // ---- Performance ----
    cout << "\n--- Performance ---\n";
    cout << fixed << setprecision(3);
    cout << "BC execution time      : " << bcTime               << " s\n";
    cout << "Total time (load + BC) : " << (loadTime + bcTime)  << " s\n";
    cout << "Peak memory (RSS)      : " << memTotal << " KB ("
         << fixed << setprecision(1) << memTotal / 1024.0       << " MB)\n";
    cout << "========================================\n";

    return 0;
}