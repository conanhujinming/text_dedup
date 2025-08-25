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

// These templates "teach" pybind11 how to automatically convert between
// Python dicts/sets and Abseil's high-performance hash containers.

namespace pybind11 { namespace detail {

// Caster for absl::flat_hash_map
template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct type_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : public map_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};

// Caster for absl::flat_hash_set
template <typename T, typename Hash, typename Eq, typename Alloc>
struct type_caster<absl::flat_hash_set<T, Hash, Eq, Alloc>>
    : public set_caster<absl::flat_hash_set<T, Hash, Eq, Alloc>, T> {};

}} // namespace pybind11::detail

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

std::string clean_utf8(std::string_view input);

// --- Helper Functions for UTF-8 Handling ---

/**
 * @brief Checks if a byte is a UTF-8 continuation byte.
 * 
 * In UTF-8, multi-byte characters consist of a leading byte followed by
 * one or more continuation bytes. Continuation bytes have the bit pattern 10xxxxxx.
 * This function efficiently checks for that pattern.
 * 
 * @param c The character (byte) to check.
 * @return True if it's a continuation byte, false otherwise.
 */
inline bool is_utf8_continuation_byte(char c) {
    // This bitmask check is equivalent to (c & 0b11000000) == 0b10000000
    return (static_cast<unsigned char>(c) & 0xC0) == 0x80;
}

/**
 * @brief Finds the next valid UTF-8 character boundary starting from a given position.
 * 
 * If the given 'pos' is in the middle of a multi-byte character, this function
 * scans forward to find the beginning of the next character. This is crucial
 * for preventing chunks from splitting a character.
 * 
 * @param text The text to scan.
 * @param pos The starting position to check.
 * @return The adjusted position that corresponds to a character boundary.
 */
size_t find_next_char_boundary(std::string_view text, size_t pos) {
    if (pos >= text.length()) {
        return text.length();
    }
    // While the current byte is a continuation byte, move forward.
    while (pos < text.length() && is_utf8_continuation_byte(text[pos])) {
        pos++;
    }
    return pos;
}


// --- Main Chunking Function (UTF-8 Aware) ---

/**
 * @brief Generates chunks and their xxHash hashes for a given text using
 *        Content-Defined Chunking (CDC), ensuring that chunks are split on
 *        UTF-8 character boundaries.
 * 
 * @param text The input text to be chunked.
 * @param min_length_dedup The target average length of a chunk, used as the divisor.
 * @param window_size The size of the rolling hash window.
 * @param generate_chunks If true, the function returns the string content of each chunk.
 *                        If false, it only returns the hashes to save memory.
 * @return A pair containing a vector of chunks (if generate_chunks is true) and
 *         a vector of their corresponding 64-bit xxHash hashes.
 */
std::pair<std::vector<std::string>, std::vector<uint64_t>>
get_chunks_and_hashes(std::string_view text, int min_length_dedup, size_t window_size = 16, bool generate_chunks = true) {
    // The divisor determines the average chunk size. Using min_length_dedup makes it adaptive.
    const uint64_t divisor = std::max(1, min_length_dedup);

    // Handle texts that are too short to be chunked.
    if (text.length() < min_length_dedup) {
        if (text.empty()) return {{}, {}};
        if (generate_chunks) {
            return {{std::string(text)}, {XXH3_64bits(text.data(), text.length())}};
        }
        return {{}, {XXH3_64bits(text.data(), text.length())}};
    }

    std::vector<std::string> chunks;
    std::vector<uint64_t> hashes;
    size_t start_pos = 0;

    // If the text is shorter than or equal to the window size, treat it as a single chunk.
    if (text.length() <= window_size) {
        if (generate_chunks) {
            chunks.emplace_back(text);
        }
        hashes.push_back(XXH3_64bits(text.data(), text.length()));
        return {std::move(chunks), std::move(hashes)};
    }

    RollingHash rh;
    rh.generate(text.substr(0, window_size));

    // Slide the window across the text.
    for (size_t i = 0; i <= text.length() - window_size; ++i) {
        size_t potential_cut_pos_end_of_window = i + window_size;
        
        // Only consider splitting if the current chunk length is at least the minimum length.
        if (potential_cut_pos_end_of_window - start_pos >= min_length_dedup) {
            // Check if the rolling hash triggers a chunk boundary.
            if ((rh.hash % divisor) == 0) {
                
                // *** CORE MODIFICATION IS HERE ***
                // Found a potential split point. Now, align it to a UTF-8 character boundary.
                size_t adjusted_cut_pos = find_next_char_boundary(text, potential_cut_pos_end_of_window);
                
                // Ensure the adjusted chunk still meets the minimum length requirement.
                if (adjusted_cut_pos - start_pos >= min_length_dedup) {
                    size_t current_chunk_length = adjusted_cut_pos - start_pos;
                    std::string_view chunk_view = {text.data() + start_pos, current_chunk_length};

                    hashes.push_back(XXH3_64bits(chunk_view.data(), chunk_view.length()));
                
                    if (generate_chunks) {
                        chunks.emplace_back(chunk_view);
                    }
                    
                    // The new chunk starts after the one we just created.
                    start_pos = adjusted_cut_pos;

                    // Optional optimization: If the boundary was adjusted significantly forward,
                    // we could jump the loop iterator 'i'. However, this would require recalculating
                    // the rolling hash. For simplicity, we let it continue sliding sequentially.
                    // i = adjusted_cut_pos - window_size;
                }
            }
        }

        // Slide the window for the next iteration.
        if (i < text.length() - window_size) {
            rh.slide(text[i], text[i + window_size]);
        }
    }

    // Handle the final remaining part of the text after the loop.
    if (start_pos < text.length()) {
        std::string_view remaining_chunk = {text.data() + start_pos, text.length() - start_pos};
        if (generate_chunks) {
            chunks.emplace_back(remaining_chunk);
        }
        hashes.push_back(XXH3_64bits(remaining_chunk.data(), remaining_chunk.length()));
    }
    
    return {std::move(chunks), std::move(hashes)};
}

// Generates CDC xxHash hashes for texts
std::vector<uint64_t> get_CDC_hashes_cpp(
    const std::vector<std::string>& texts, // Use std::string for safety in bindings
    int min_length_dedup, 
    size_t window_size = 16
) {
    std::vector<uint64_t> results;
    // Optional: Pre-allocate a reasonable amount of memory if you can estimate the total number of hashes
    // results.reserve(texts.size() * AVERAGE_HASHES_PER_TEXT);

    for (const auto& text : texts) {
        // We only care about the hashes, not the chunks.
        auto [_, hashes] = get_chunks_and_hashes(text, min_length_dedup, window_size, false);
        
        if (!hashes.empty()) {
            // This is the key optimization:
            // Insert all elements from the `hashes` vector into the end of the `results` vector.
            results.insert(results.end(), hashes.begin(), hashes.end());
        }
    }
    return results; // No need for std::move on return for modern compilers (RVO)
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
        const __m256i ones = _mm256_set1_epi32(1);

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
            // The original logic incorrectly used the sign bit. The corrected logic uses the
            // lowest bit (`& 1`), making it consistent with the scalar fallback path.

            // 1. Isolate the lowest bit of each 32-bit hash. Each lane becomes 0 or 1.
            __m256i lowest_bits = _mm256_and_si256(prng_hash, ones);

            // 2. Create a mask. If a lane in lowest_bits is 1, the corresponding lane
            //    in the mask becomes all 1s (0xFFFFFFFF). Otherwise, it's all 0s.
            __m256i mask = _mm256_cmpeq_epi32(lowest_bits, ones);
            
            // 3. Blend using the mask. If mask bit is 1 (i.e., hash's lowest bit was 1),
            //    select from weights_add. Otherwise, select from weights_sub.
            //    _mm256_blendv_epi8 selects from the first operand if the mask's high bit is 1.
            __m256i delta = _mm256_blendv_epi8(weights_add, weights_sub, mask);
            // --- BUG FIX ENDS HERE ---
            
            __m256i v_new = _mm256_add_epi32(v_current, delta);
            _mm256_storeu_si256((__m256i*)&v[i], v_new);
        }

        // Handle the remainder (This part was already correct)
        for (; i < hashbits; ++i) {
            if (XXH3_64bits_withSeed(&feature_hash, sizeof(feature_hash), i) & 1) {
                v[i] += weight;
            } else {
                v[i] -= weight;
            }
        }
    }

#else
    // Fallback scalar path (This part was already correct)
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
std::string clean_utf8(std::string_view input) {
    std::string output;
    output.reserve(input.length());

    const unsigned char* p = (const unsigned char*)input.data();
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

// Template parameter `IDType` can be `int` or `std::string`.
template<typename IDType>
absl::flat_hash_set<IDType> cluster_and_find_duplicates(
    const std::vector<uint8_t>& binary_vectors,
    const std::vector<IDType>& doc_ids, // Maps Faiss index (0,1,2...) to original ID
    int hamming_threshold,
    const std::string& faiss_index_type,
    int simhash_bits)
{
    size_t num_docs = doc_ids.size();
    if (num_docs == 0) {
        return {};
    }

    std::unique_ptr<faiss::IndexBinary> index;

    if (faiss_index_type == "flat") {
        index = std::make_unique<faiss::IndexBinaryFlat>(simhash_bits);
    } else if (faiss_index_type == "hash") {
        index = std::make_unique<faiss::IndexBinaryHash>(simhash_bits, 64);
    } else if (faiss_index_type == "IVF") {
        int nlist = static_cast<int>(4 * std::sqrt(num_docs));
        nlist = std::max(1, std::min(nlist, 65536));
        auto quantizer = new faiss::IndexBinaryFlat(simhash_bits);
        index = std::make_unique<faiss::IndexBinaryIVF>(quantizer, simhash_bits, nlist);
    } else { // Note: HNSW is not ideal for range search, so we can omit it as a primary option
        index = std::make_unique<faiss::IndexBinaryFlat>(simhash_bits);
    }

    if (!index->is_trained) {
        index->train(num_docs, binary_vectors.data());
    }
    index->add(num_docs, binary_vectors.data());
    
    if (auto ivf_index = dynamic_cast<faiss::IndexBinaryIVF*>(index.get())) {
        ivf_index->nprobe = std::min((int)ivf_index->nlist, 16);
    }

    faiss::RangeSearchResult res(num_docs);
    index->range_search(num_docs, binary_vectors.data(), hamming_threshold, &res);

    // --- 2. Find Connected Components / Clusters ---
    UnionFind uf(num_docs);
    for (size_t i = 0; i < num_docs; ++i) {
        for (size_t j = res.lims[i]; j < res.lims[i + 1]; ++j) {
            uf.unite(i, res.labels[j]);
        }
    }

    absl::flat_hash_map<int, std::vector<size_t>> components;
    for (size_t i = 0; i < num_docs; ++i) {
        components[uf.find(i)].push_back(i);
    }

    // --- 3. Identify Documents to Remove ---
    absl::flat_hash_set<IDType> to_remove;
    for (auto const& [root, component_indices] : components) {
        if (component_indices.size() > 1) {
            // Find the document to keep. The rule is to keep the "smallest" ID.
            // This works for both integers and lexicographically for strings.
            const IDType* id_to_keep = &doc_ids[component_indices[0]];
            for (size_t i = 1; i < component_indices.size(); ++i) {
                if (doc_ids[component_indices[i]] < *id_to_keep) {
                    id_to_keep = &doc_ids[component_indices[i]];
                }
            }
            
            // Add all other documents in the cluster to the removal set.
            for (size_t idx : component_indices) {
                if (doc_ids[idx] != *id_to_keep) {
                    to_remove.insert(doc_ids[idx]);
                }
            }
        }
    }
    
    return to_remove;
}

absl::flat_hash_set<std::string> find_duplicates_cpp(
    const absl::flat_hash_map<std::string, std::vector<uint64_t>>& all_signatures,
    int hamming_threshold,
    const std::string& faiss_index_type,
    int simhash_bits
)
{
    if (all_signatures.empty()) return {};

    // --- 1. Prepare data for the helper function ---
    std::vector<std::string> doc_ids;
    doc_ids.reserve(all_signatures.size());
    for (const auto& [id, sig] : all_signatures) {
        doc_ids.push_back(id);
    }
    std::sort(doc_ids.begin(), doc_ids.end());

    size_t num_docs = doc_ids.size();
    int hash_bytes = simhash_bits / 8;
    std::vector<uint8_t> binary_vectors(num_docs * hash_bytes);

    for (size_t i = 0; i < num_docs; ++i) {
        const auto& signature_vec = all_signatures.at(doc_ids[i]);
        memcpy(&binary_vectors[i * hash_bytes], signature_vec.data(), hash_bytes);
    }

    // --- 2. Call the helper function and return its result ---
    absl::flat_hash_set<std::string> to_remove = cluster_and_find_duplicates<std::string>(
        binary_vectors,
        doc_ids, // `doc_ids` is our std::vector<std::string>
        hamming_threshold,
        faiss_index_type,
        simhash_bits
    );

    std::cout << "--- C++ Core: Found " << to_remove.size() << " near-duplicate documents to remove ---" << std::endl;
    return to_remove;
}

struct ClientSignatures {
    absl::flat_hash_map<std::string, std::vector<uint64_t>> doc_signatures;
};

// This function takes a whole batch of texts and processes them inside C++
// to avoid the slow Python for-loop.
ClientSignatures process_texts_for_signatures_cpp(
    const std::vector<std::string>& texts,
    const std::string& client_id_str,
    int min_length_dedup,
    int simhash_bits // Corresponds to hash_permutations in your script
)
{
    ClientSignatures result;
    
    // We can parallelize this loop inside C++ for maximum performance!
    // We need thread-local storage for the results to avoid locking.
    std::vector<absl::flat_hash_map<std::string, std::vector<uint64_t>>> local_doc_signatures(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < texts.size(); ++i) {
        int thread_id = omp_get_thread_num();
        const auto& text = texts[i];
        
        // 1. Get document signature
        
        std::string global_doc_id = client_id_str + "_" + std::to_string(i);
        std::vector<uint64_t> sig = get_document_simhash_performance(text, simhash_bits);
        
        // Assuming non-empty signatures are valid
        bool is_valid_sig = false;
        for(uint64_t val : sig) {
            if (val != 0) {
                is_valid_sig = true;
                break;
            }
        }
        if (is_valid_sig) {
            local_doc_signatures[thread_id][global_doc_id] = std::move(sig);
        }
    }

    // Merge results from all threads into the final result object
    for (const auto& local_map : local_doc_signatures) {
        result.doc_signatures.insert(local_map.begin(), local_map.end());
    }

    return result;
}

// This function takes a batch of original texts and a set of "bad" chunk hashes.
// It performs the CDC cleaning process for all texts in parallel within C++
// and returns a vector of the cleaned texts.
std::vector<std::string> clean_texts_with_bad_hashes_cpp(
    const std::vector<std::string>& original_texts,
    const absl::flat_hash_set<uint64_t>& bad_chunk_hashes,
    int min_length_dedup,
    size_t window_size = 16)
{
    std::vector<std::string> cleaned_texts(original_texts.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < original_texts.size(); ++i) {
        const auto& text = original_texts[i];
        
        // Step 1: Get the chunks and hashes for the original text.
        auto [chunks, hashes] = get_chunks_and_hashes(text, min_length_dedup, window_size);
        
        // Step 2: Filter out the chunks whose hashes are in the bad set.
        std::string cleaned_text;
        // Pre-allocate memory to avoid reallocations. This is a safe upper bound.
        cleaned_text.reserve(text.length()); 

        for (size_t j = 0; j < chunks.size(); ++j) {
            // `contains` on absl::flat_hash_set is very fast.
            if (bad_chunk_hashes.find(hashes[j]) == bad_chunk_hashes.end()) {
                // If the hash is NOT bad, keep the chunk.
                // Use std::move to efficiently append.
                cleaned_text.append(std::move(chunks[j]));
            }
        }
        
        // Step 3: Assign the reconstructed, cleaned text to the result vector.
        cleaned_texts[i] = std::move(cleaned_text);
    }
    
    return cleaned_texts;
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
    std::vector<std::vector<std::pair<uint64_t, std::string>>> doc_chunks_info(docs.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < docs.size(); ++i) {
        // `get_chunks_and_hashes` correctly returns std::vector<std::string>
        auto [chunks, hashes] = get_chunks_and_hashes(clean_utf8(docs[i]), min_length_dedup);
        
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
    // --- Stage 3: Faiss Near-Duplicate Detection (now refactored) ---
    std::cout << "--- Stage 3: C++ Faiss Near-Duplicate Detection ---" << std::endl;
    absl::flat_hash_set<int> to_remove; // Note: type is `int`

    if (!valid_indices.empty()) {
        size_t num_valid_docs = valid_indices.size();
        int hash_bytes = simhash_bits / 8;
        std::vector<uint8_t> binary_vectors(num_valid_docs * hash_bytes);

        for (size_t i = 0; i < num_valid_docs; ++i) {
            memcpy(&binary_vectors[i * hash_bytes], signatures[valid_indices[i]].data(), hash_bytes);
        }

        to_remove = cluster_and_find_duplicates<int>(
            binary_vectors,
            valid_indices, // `valid_indices` is our std::vector<int>
            hamming_threshold,
            faiss_index_type,
            simhash_bits
        );
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
    m.def("get_CDC_hashes_cpp",
        &get_CDC_hashes_cpp, // Direct function pointer
        "Calculate CDC hashes",
        py::arg("texts"),
        py::arg("min_length_dedup"),
        py::arg("window_size")
    );
    m.def("get_chunks_and_hashes",
        &get_chunks_and_hashes, // Direct function pointer
        "Calculate Chunks and hashes",
        py::arg("texts"),
        py::arg("min_length_dedup"),
        py::arg("window_size"),
        py::arg("generate_chunks")
    );
    m.def("find_duplicates_cpp",
        &find_duplicates_cpp, // Direct function pointer
        "Find duplicates",
        py::arg("all_signatures"),
        py::arg("hamming_threshold"),
        py::arg("faiss_index_type"),
        py::arg("simhash_bits")
    );
    m.def("get_document_simhash_performance",
        &get_document_simhash_performance, // Direct function pointer
        "get sim hash",
        py::arg("text"),
        py::arg("hashbits")
    );
    // Bind the result struct
    py::class_<ClientSignatures>(m, "ClientSignatures")
        .def(py::init<>())
        .def_readonly("doc_signatures", &ClientSignatures::doc_signatures);

    // Bind the new batch processing function
    m.def("process_texts_for_signatures_cpp",
        &process_texts_for_signatures_cpp,
        "Processes a batch of texts to extract chunk hashes and document signatures in C++.",
        py::arg("texts"),
        py::arg("client_id_str"),
        py::arg("min_length_dedup"),
        py::arg("simhash_bits")
    );

    m.def("clean_texts_with_bad_hashes_cpp",
        &clean_texts_with_bad_hashes_cpp,
        "Cleans a batch of texts by removing chunks specified in a 'bad hash' set.",
        py::arg("original_texts"),
        py::arg("bad_chunk_hashes"),
        py::arg("min_length_dedup"),
        py::arg("window_size") = 16
    );
}