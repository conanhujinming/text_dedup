#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <tuple>

#include <omp.h> // For parallelization
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/ResultHandler.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

// --- BEGIN xxHash Integration ---
// This macro ensures that function symbols are defined as static,
// preventing linkage errors in a shared library.
#define XXH_STATIC_LINKING_ONLY
// This macro triggers the inclusion of the C implementation file
// directly into this compilation unit.
#define XXH_IMPLEMENTATION
#include "xxhash.h"
// --- END xxHash Integration ---

// Include header for AVX2 intrinsics
#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

namespace py = pybind11;

// A simple C++ implementation of RollingHash
struct RollingHash {
    uint64_t base = 257;
    uint64_t prime = 1000000007;
    uint64_t hash = 0;
    uint64_t power = 1;
    size_t window_size = 0;

    void generate(std::string_view window) {
        window_size = window.length();
        power = 1;
        for (size_t i = 0; i < window_size - 1; ++i) {
            power = (power * base) % prime;
        }
        for (char c : window) {
            hash = (hash * base + static_cast<unsigned char>(c)) % prime;
        }
    }

    void slide(char old_char, char new_char) {
        hash = (hash + prime - (static_cast<unsigned char>(old_char) * power) % prime) % prime;
        hash = (hash * base + static_cast<unsigned char>(new_char)) % prime;
    }
};


// Generates chunks and their xxHash hashes for a given text
std::pair<std::vector<std::string_view>, std::vector<uint64_t>>
get_chunks_and_hashes(std::string_view text, int min_length_dedup, size_t window_size = 16) {
    const uint64_t divisor = std::max(1, min_length_dedup);

    if (text.length() < min_length_dedup) {
        if (text.empty()) return {{}, {}};
        return {{std::string_view(text)}, {XXH3_64bits(text.data(), text.length())}};
    }

    std::vector<std::string_view> chunks; // Changed from string_view to string
    std::vector<uint64_t> hashes;
    size_t start_pos = 0;

    if (text.length() <= window_size) {
        chunks.emplace_back(text);
        hashes.push_back(XXH3_64bits(text.data(), text.length()));
        return {std::move(chunks), std::move(hashes)};
    }

    RollingHash rh;
    rh.generate(text.substr(0, window_size));

    for (size_t i = 0; i <= text.length() - window_size; ++i) {
        size_t current_pos_end_of_window = i + window_size;
        size_t current_chunk_length = current_pos_end_of_window - start_pos;
        
        if (current_chunk_length >= min_length_dedup) {
            if ((rh.hash % divisor) == 0) {
                // Directly create a std::string from the substring view
                chunks.emplace_back(text.data() + start_pos, current_chunk_length);
                start_pos = current_pos_end_of_window;
            }
        }

        if (i < text.length() - window_size) {
            rh.slide(text[i], text[i + window_size]);
        }
    }

    if (start_pos < text.length()) {
        chunks.emplace_back(text.data() + start_pos, text.length() - start_pos);
    }
    
    hashes.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        // Now `chunk` is a std::string, so chunk.data() is safe
        hashes.push_back(XXH3_64bits(chunk.data(), chunk.length()));
    }
    return {std::move(chunks), std::move(hashes)};
}

// Generates a SimHash for a document
std::vector<uint64_t> get_document_simhash(std::string_view text, int hashbits) {
    const int num_blocks = (hashbits + 63) / 64;
    std::vector<uint64_t> fingerprint(num_blocks, 0);

    std::unordered_map<std::string, int> features;
    std::string word;
    for (char c : text) {
        if (std::isalnum(c)) {
            word += std::tolower(c);
        } else if (!word.empty()) {
            features[word]++;
            word.clear();
        }
    }
    if (!word.empty()) features[word]++;

    if (features.empty()) return fingerprint;

    std::vector<int> v(hashbits, 0);
    // Use a more robust hash for features to avoid collisions
    std::hash<std::string> hasher;

    for (const auto& [feature, weight] : features) {
        uint64_t h = hasher(feature);
        for (int i = 0; i < hashbits; ++i) {
            // Using a simple PRNG-like step to generate different bits from the same hash
            uint64_t bit_hash = h;
            bit_hash ^= bit_hash << 13;
            bit_hash ^= bit_hash >> 7;
            bit_hash ^= bit_hash << 17;
            bit_hash += i; // Add offset for the bit position
            if ((bit_hash >> (i % 64)) & 1) v[i] += weight;
            else v[i] -= weight;
        }
    }

    for (int i = 0; i < hashbits; ++i) {
        if (v[i] >= 0) {
            fingerprint[i / 64] |= (1ULL << (i % 64));
        }
    }
    return fingerprint;
}

// A high-performance implementation of SimHash generation.
// It uses AVX2 for vectorization if available, otherwise falls back to a fast scalar version.
std::vector<uint64_t> get_document_simhash_performance(std::string_view text, int hashbits) {
    const int num_blocks = (hashbits + 63) / 64;
    std::vector<uint64_t> fingerprint(num_blocks, 0);

    // Stage 1: Fast, zero-copy feature counting
    std::string lower_text;
    lower_text.reserve(text.length());
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            lower_text += std::tolower(static_cast<unsigned char>(c));
        } else {
            lower_text += ' ';
        }
    }
    
    absl::flat_hash_map<std::string_view, int> features;
    size_t start = 0;
    for (size_t i = 0; i <= lower_text.length(); ++i) {
        if (i == lower_text.length() || lower_text[i] == ' ') {
            if (i > start) {
                features[std::string_view(lower_text.data() + start, i - start)]++;
            }
            start = i + 1;
        }
    }

    if (features.empty()) {
        return fingerprint;
    }

    // Stage 2: Calculate the weighted vector 'v'
    std::vector<int32_t> v(hashbits, 0);
    
#ifdef __AVX2__
    // AVX2-optimized path: process 8 integers at a time
    for (const auto& [feature, weight] : features) {
        uint64_t feature_hash = XXH3_64bits(feature.data(), feature.length());

        const __m256i weights_add = _mm256_set1_epi32(weight);
        const __m256i weights_sub = _mm256_set1_epi32(-weight);

        int i = 0;
        for (; i <= hashbits - 8; i += 8) {
            __m256i prng_hash = _mm256_set_epi32(
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 7),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 6),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 5),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 4),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 3),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 2),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 1),
                XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i + 0)
            );

            __m256i v_current = _mm256_loadu_si256((__m256i*)&v[i]);
            
            // --- BUG FIX STARTS HERE ---
            // Create a proper mask. If the sign bit of a 32-bit integer in prng_hash is 1,
            // the corresponding 32-bit lane in 'mask' will be all 1s (0xFFFFFFFF).
            // Otherwise, it will be all 0s (0x00000000).
            __m256i mask = _mm256_srai_epi32(prng_hash, 31);
            
            // Now, blendv will work correctly because all bytes within a 32-bit lane
            // of the mask will have the same sign bit.
            __m256i delta = _mm256_blendv_epi8(weights_sub, weights_add, mask);
            // --- BUG FIX ENDS HERE ---
            
            __m256i v_new = _mm256_add_epi32(v_current, delta);
            _mm256_storeu_si256((__m256i*)&v[i], v_new);
        }

        // Handle the remainder
        for (; i < hashbits; ++i) {
            if (XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i) & 1) {
                v[i] += weight;
            } else {
                v[i] -= weight;
            }
        }
    }

#else
    // Fallback scalar path (this part was already correct)
    for (const auto& [feature, weight] : features) {
        uint64_t feature_hash = XXH3_64bits(feature.data(), feature.length());
        for (int i = 0; i < hashbits; ++i) {
            if (XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i) & 1) {
                v[i] += weight;
            } else {
                v[i] -= weight;
            }
        }
    }
#endif

    // Stage 3: Finalize the fingerprint
    for (int i = 0; i < hashbits; ++i) {
        if (v[i] >= 0) {
            fingerprint[i / 64] |= (1ULL << (i % 64));
        }
    }
    return fingerprint;
}

// A simple Union-Find (Disjoint Set Union) data structure for clustering
struct UnionFind {
    std::vector<int> parent;
    UnionFind(int n) {
        parent.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
        }
    }
};

// Function to validate and clean a UTF-8 string.
// It removes invalid byte sequences.
std::string clean_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    const unsigned char* p = (const unsigned char*)input.c_str();
    const unsigned char* end = p + input.length();

    while (p < end) {
        // Single-byte character (ASCII)
        if (*p < 0x80) {
            output += *p++;
            continue;
        }
        // Multi-byte character
        int len = 0;
        if ((*p & 0xE0) == 0xC0) len = 2;      // 2-byte sequence
        else if ((*p & 0xF0) == 0xE0) len = 3; // 3-byte sequence
        else if ((*p & 0xF8) == 0xF0) len = 4; // 4-byte sequence
        else {
            // Invalid start byte, skip it
            p++;
            continue;
        }

        // Check if the sequence is complete
        if (p + len > end) {
            // Incomplete sequence at the end of the string, drop it
            break;
        }

        // Check if continuation bytes are valid
        bool valid = true;
        for (int i = 1; i < len; ++i) {
            if ((p[i] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            // Append the valid sequence
            output.append((const char*)p, len);
            p += len;
        } else {
            // Invalid sequence, skip the start byte
            p++;
        }
    }
    return output;
}


// The main deduplication function
std::vector<std::optional<std::string>> deduplicate_cpp(
    const std::vector<std::string>& docs,
    int min_length_dedup,
    int hamming_threshold,
    const std::string& faiss_index_type,
    int simhash_bits) 
{
    // --- Stage 1: Parallel CDC Deduplication ---
    std::cout << "--- Stage 1: C++ Parallel CDC Deduplication ---" << std::endl;
    
    // 1.A: Generate chunks and hashes, ensuring data ownership.
    
    // === BUG FIX 1: Change the type to std::string to own the data ===
    std::vector<std::vector<std::pair<uint64_t, std::string_view>>> doc_chunks_info(docs.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < docs.size(); ++i) {
        // `get_chunks_and_hashes` correctly returns std::vector<std::string>
        auto [chunks, hashes] = get_chunks_and_hashes(docs[i], min_length_dedup);
        
        doc_chunks_info[i].reserve(chunks.size());
        for (size_t j = 0; j < chunks.size(); ++j) {
            // === BUG FIX 2: Use std::move to efficiently transfer ownership ===
            // Move the string from the local `chunks` vector into `doc_chunks_info`.
            // This is very efficient and avoids copies.
            doc_chunks_info[i].emplace_back(hashes[j], std::move(chunks[j]));
        }
    }

    // 1.B: Find the first occurrence of each unique chunk hash.
    absl::flat_hash_set<uint64_t> global_seen_hashes;
    
    // This can still be string_view, as the data it views is now safely owned by `doc_chunks_info`.
    std::vector<std::vector<std::string_view>> chunks_to_keep_per_doc(docs.size());
    
    for(size_t i = 0; i < doc_chunks_info.size(); ++i) {
        for (const auto& [hash, chunk_str] : doc_chunks_info[i]) { // chunk_str is a std::string
            if (global_seen_hashes.insert(hash).second) {
                // An std::string_view is created here. It safely points to the data
                // inside `doc_chunks_info[i]`, which will exist for the whole function's lifetime.
                chunks_to_keep_per_doc[i].push_back(chunk_str);
            }
        }
    }

    // 1.C: Reconstruct the documents.
    std::vector<std::string> deduped_texts(docs.size());
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < docs.size(); ++i) {
        size_t total_len = 0;
        for (const auto& sv : chunks_to_keep_per_doc[i]) {
            total_len += sv.length();
        }
        std::string result_text;
        result_text.reserve(total_len);
        for (const auto& sv : chunks_to_keep_per_doc[i]) {
            result_text.append(sv);
        }
        deduped_texts[i] = std::move(result_text);
    }
    
    // =======================================================================
    // === START: ADD DIAGNOSTIC CODE HERE ===
    // =======================================================================
    
    // Calculate the total size of data before and after CDC deduplication.
    // This requires iterating over the original `docs` and the new `deduped_texts`.
    // We use long long to avoid overflow with large datasets.
    long long original_size_bytes = 0;
    long long deduped_size_bytes = 0;

    // This can be parallelized for very large document counts, but for logging,
    // a serial loop is usually fast enough and simpler.
    #pragma omp parallel for reduction(+:original_size_bytes, deduped_size_bytes)
    for (size_t i = 0; i < docs.size(); ++i) {
        original_size_bytes += docs[i].length();
        deduped_size_bytes += deduped_texts[i].length();
    }
    
    // Calculate the reduction percentage.
    double reduction_ratio = 0.0;
    if (original_size_bytes > 0) {
        reduction_ratio = 100.0 * (1.0 - static_cast<double>(deduped_size_bytes) / static_cast<double>(original_size_bytes));
    }
    
    // Print the statistics to the console.
    std::cout << "--- Stage 1 Diagnostics ---" << std::endl;
    std::cout << "Original data size: " << original_size_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Data size after CDC: " << deduped_size_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "CDC removed: " << (original_size_bytes - deduped_size_bytes) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "CDC reduction ratio: " << reduction_ratio << "%" << std::endl;
    
    // =======================================================================
    // === END: ADD DIAGNOSTIC CODE HERE ===
    // =======================================================================

    // --- Stage 2: Parallel SimHash Signature Generation (lock-free) ---
    std::cout << "--- Stage 2: C++ Parallel SimHash Generation ---" << std::endl;
    const int num_hash_blocks = (simhash_bits + 63) / 64;
    std::vector<std::vector<uint64_t>> signatures(docs.size());
    std::vector<int> valid_indices;
    
    // Use thread-local storage to collect valid indices without locking
    std::vector<std::vector<int>> local_valid_indices(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < deduped_texts.size(); ++i) {
        if (!deduped_texts[i].empty()) {
            // signatures[i] = get_document_simhash(deduped_texts[i], simhash_bits);
            signatures[i] = get_document_simhash_performance(deduped_texts[i], simhash_bits);
            local_valid_indices[omp_get_thread_num()].push_back(i);
        }
    }

    // Merge the results from each thread
    for(const auto& local_vec : local_valid_indices) {
        valid_indices.insert(valid_indices.end(), local_vec.begin(), local_vec.end());
    }
    std::sort(valid_indices.begin(), valid_indices.end());

    // --- Stage 3: Faiss Near-Duplicate Detection ---
    std::cout << "--- Stage 3: C++ Faiss Near-Duplicate Detection ---" << std::endl;
    if (valid_indices.empty()) {
        return {};
    }

    size_t num_valid_docs = valid_indices.size();
    int hash_bytes = simhash_bits / 8;
    std::vector<uint8_t> binary_vectors(num_valid_docs * hash_bytes);

    for (size_t i = 0; i < num_valid_docs; ++i) {
        memcpy(&binary_vectors[i * hash_bytes], signatures[valid_indices[i]].data(), hash_bytes);
    }
    
    std::unique_ptr<faiss::IndexBinary> index;
    // Faiss index creation logic
    int nlist = 0;
    if (faiss_index_type == "flat") {
        index = std::make_unique<faiss::IndexBinaryFlat>(simhash_bits);
    } else if (faiss_index_type == "hash") {
        int n_hash_tables = 64;
        index = std::make_unique<faiss::IndexBinaryHash>(simhash_bits, n_hash_tables);
    } else if (faiss_index_type == "IVF") {
        if (num_valid_docs < 10000) {
            nlist = std::max(1, (int)num_valid_docs / 16);
        } else {
            // A common heuristic for nlist
            nlist = static_cast<int>(4 * std::sqrt(num_valid_docs));
        }
        nlist = std::min(nlist, 65536); 
        nlist = std::max(nlist, 1);
        auto quantizer = new faiss::IndexBinaryFlat(simhash_bits);
        index = std::make_unique<faiss::IndexBinaryIVF>(quantizer, simhash_bits, nlist);
    } else if (faiss_index_type == "HNSW") {
        int M = 32;
        index = std::make_unique<faiss::IndexBinaryHNSW>(simhash_bits, M);
    } else {
        std::cerr << "Warning: Unknown Faiss index type '" << faiss_index_type 
                  << "'. Falling back to 'flat'." << std::endl;
        index = std::make_unique<faiss::IndexBinaryFlat>(simhash_bits);
    }

    if (!index->is_trained) {
        std::cout << "Training Faiss " << faiss_index_type << " index..." << std::endl;
        // Use a subset for training if the dataset is large
        size_t train_size = std::min(num_valid_docs, (size_t)256 * 100);
        index->train(train_size, binary_vectors.data());
    }

    std::cout << "Adding " << num_valid_docs << " vectors to Faiss index..." << std::endl;
    index->add(num_valid_docs, binary_vectors.data());

    faiss::RangeSearchResult res(num_valid_docs);
    if (auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index.get())) {
        ivf_index->nprobe = std::min(nlist, 16); // Start with a reasonable nprobe
        std::cout << "Faiss IVF using nprobe = " << ivf_index->nprobe << std::endl;
    }
    index->range_search(num_valid_docs, binary_vectors.data(), hamming_threshold, &res);

    // --- Find connected components using Union-Find ---
    UnionFind uf(num_valid_docs);
    for (size_t i = 0; i < num_valid_docs; ++i) {
        for (size_t j = res.lims[i]; j < res.lims[i+1]; ++j) {
            // No need to check for i != res.labels[j], unite is idempotent
            uf.unite(i, res.labels[j]);
        }
    }

    // Group documents by their component root
    absl::flat_hash_map<int, std::vector<int>> components;
    for (size_t i = 0; i < num_valid_docs; ++i) {
        // Map Faiss index (i) back to original doc index
        components[uf.find(i)].push_back(valid_indices[i]);
    }
    
    std::unordered_set<int> to_remove;
    for (auto const& [root, component_indices] : components) {
        if (component_indices.size() > 1) {
            // Keep the document with the smallest original index
            auto min_it = std::min_element(component_indices.begin(), component_indices.end());
            for (int doc_idx : component_indices) {
                if (doc_idx != *min_it) {
                    to_remove.insert(doc_idx);
                }
            }
        }
    }
    
    std::vector<std::optional<std::string>> final_results;
    final_results.reserve(docs.size());

    // First, populate the vector with all the cleaned texts.
    for (auto& text : deduped_texts) {
        final_results.emplace_back(clean_utf8(text));
    }
    
    // Now, iterate through the indices that need to be removed
    // and replace their corresponding entries with std::nullopt (which becomes None in Python).
    for (int idx : to_remove) {
        if (idx < final_results.size()) {
            final_results[idx] = std::nullopt;
        }
    }
    return final_results;

}


PYBIND11_MODULE(_core, m) {
    m.doc() = "High-performance C++ deduplication module for Python";
    m.def("deduplicate_cpp",
        &deduplicate_cpp,
        "Performs CDC and SimHash deduplication.",
        py::arg("docs"),
        py::arg("min_length_dedup"),
        py::arg("hamming_threshold"),
        py::arg("faiss_index_type"),
        py::arg("simhash_bits") = 64
    );
    m.def("get_chunks_and_hashes",
        &get_chunks_and_hashes, // Direct function pointer
        "Calculate chunks and hashes",
        py::arg("text"),
        py::arg("min_length_dedup"),
        py::arg("window_size")
    );
}