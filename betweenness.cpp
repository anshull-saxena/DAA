#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace std;

struct Graph {
    vector<vector<int>> adj;
    vector<int> reverse_ids;
    long long edge_count = 0;
};

struct Result {
    vector<pair<int, double>> top_nodes;
    double execution_seconds = 0.0;
    double memory_usage_mb = 0.0;
    double analytical_memory_mb = 0.0;
};

static double bytesToMB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double readStatusValueKB(const string& key) {
#if defined(__linux__)
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line)) {
        if (line.rfind(key, 0) == 0) {
            istringstream iss(line);
            string label;
            double value_kb = 0.0;
            string unit;
            iss >> label >> value_kb >> unit;
            return value_kb;
        }
    }
#else
    (void)key;
#endif
    return 0.0;
}

static double currentRSSMB() {
    double kb = readStatusValueKB("VmRSS:");
    if (kb > 0.0) {
        return kb / 1024.0;
    }

#if defined(__linux__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
#endif
    return 0.0;
}

static size_t estimateMemoryBytes(size_t n, long long m) {
    size_t total = 0;
    total += sizeof(vector<int>) * n;                  // adjacency list headers
    total += sizeof(int) * static_cast<size_t>(m);     // adjacency edges
    total += sizeof(vector<int>) * n;                  // predecessor list headers
    total += sizeof(int) * static_cast<size_t>(m);     // predecessor edges (worst-case)
    total += sizeof(int) * n;                          // reverse id map
    total += sizeof(long long) * n;                    // sigma
    total += sizeof(int) * n;                          // distance
    total += sizeof(double) * n;                       // delta
    total += sizeof(double) * n;                       // betweenness
    total += sizeof(int) * n;                          // BFS/stack storage
    total += sizeof(int) * n;                          // queue storage
    total += sizeof(int) * n;                          // visited vertices
    return total;
}

static Graph loadGraph(const string& filename) {
    ifstream input(filename);
    if (!input) {
        throw runtime_error("Failed to open input file: " + filename);
    }

    Graph graph;
    unordered_map<int, int> id_to_index;
    id_to_index.reserve(1 << 20);

    string line;
    while (getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        istringstream iss(line);
        int from = 0;
        int to = 0;
        if (!(iss >> from >> to)) {
            continue;
        }

        auto mapNode = [&](int original_id) -> int {
            auto it = id_to_index.find(original_id);
            if (it != id_to_index.end()) {
                return it->second;
            }

            int new_index = static_cast<int>(graph.adj.size());
            id_to_index.emplace(original_id, new_index);
            graph.adj.emplace_back();
            graph.reverse_ids.push_back(original_id);
            return new_index;
        };

        int u = mapNode(from);
        int v = mapNode(to);
        graph.adj[u].push_back(v);
        ++graph.edge_count;
    }

    return graph;
}

static Result computeBetweenness(const Graph& graph) {
    const int n = static_cast<int>(graph.adj.size());
    vector<double> cb(n, 0.0);
    vector<long long> sigma(n, 0);
    vector<int> dist(n, -1);
    vector<double> delta(n, 0.0);
    vector<vector<int>> predecessors(n);
    vector<int> stack_order;
    vector<int> visited;

    stack_order.reserve(n);
    visited.reserve(n);

    const double rss_before = currentRSSMB();
    const auto start = chrono::steady_clock::now();

    for (int source = 0; source < n; ++source) {
        queue<int> bfs_queue;
        stack_order.clear();
        visited.clear();

        sigma[source] = 1;
        dist[source] = 0;
        visited.push_back(source);
        bfs_queue.push(source);

        while (!bfs_queue.empty()) {
            int v = bfs_queue.front();
            bfs_queue.pop();
            stack_order.push_back(v);

            for (int w : graph.adj[v]) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    bfs_queue.push(w);
                    visited.push_back(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    predecessors[w].push_back(v);
                }
            }
        }

        for (auto it = stack_order.rbegin(); it != stack_order.rend(); ++it) {
            int w = *it;
            for (int v : predecessors[w]) {
                if (sigma[w] != 0) {
                    delta[v] += (static_cast<double>(sigma[v]) / static_cast<double>(sigma[w])) * (1.0 + delta[w]);
                }
            }
            if (w != source) {
                cb[w] += delta[w];
            }
        }

        for (int v : visited) {
            sigma[v] = 0;
            dist[v] = -1;
            delta[v] = 0.0;
            predecessors[v].clear();
        }
    }

    const auto end = chrono::steady_clock::now();
    const double rss_after = currentRSSMB();

    vector<int> order(n);
    for (int i = 0; i < n; ++i) {
        order[i] = i;
    }

    sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (cb[lhs] == cb[rhs]) {
            return graph.reverse_ids[lhs] < graph.reverse_ids[rhs];
        }
        return cb[lhs] > cb[rhs];
    });

    Result result;
    const int limit = min(20, n);
    result.top_nodes.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        int idx = order[i];
        result.top_nodes.emplace_back(graph.reverse_ids[idx], cb[idx]);
    }

    result.execution_seconds = chrono::duration<double>(end - start).count();
    result.analytical_memory_mb = bytesToMB(estimateMemoryBytes(static_cast<size_t>(n), graph.edge_count));

    double rss_delta = 0.0;
    if (rss_before > 0.0 && rss_after >= rss_before) {
        rss_delta = rss_after - rss_before;
    } else if (rss_after > 0.0) {
        rss_delta = rss_after;
    }

    result.memory_usage_mb = max(rss_delta, result.analytical_memory_mb);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <edge_list_file>" << '\n';
        return 1;
    }

    const string filename = argv[1];

    try {
        Graph graph = loadGraph(filename);
        Result result = computeBetweenness(graph);

        cout << "========================================" << '\n';
        cout << "Dataset: " << filename << '\n';
        cout << "========================================" << '\n';
        cout << "Graph loaded: " << graph.adj.size() << " nodes, " << graph.edge_count << " edges" << '\n';
        cout << '\n';
        cout << "--- Top 20 Nodes by Betweenness Centrality ---" << '\n';
        cout << left << setw(7) << "Rank"
             << setw(14) << "Node ID"
             << "Betweenness Score" << '\n';

        cout << fixed << setprecision(6);
        for (size_t i = 0; i < result.top_nodes.size(); ++i) {
            cout << left << setw(7) << (i + 1)
                 << setw(14) << result.top_nodes[i].first
                 << result.top_nodes[i].second << '\n';
        }

        cout << '\n';
        cout << "--- Performance ---" << '\n';
        cout << fixed << setprecision(3)
             << "Execution time : " << result.execution_seconds << " seconds" << '\n';
        cout << fixed << setprecision(2)
             << "Memory usage   : " << result.memory_usage_mb << " MB" << '\n';
        cout << "========================================" << '\n';
    } catch (const exception& ex) {
        cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
