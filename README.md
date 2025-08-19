# High-Performance Text Deduplication Toolkit

This toolkit provides a highly optimized solution for large-scale text deduplication. It employs a multi-stage pipeline that combines exact substring deduplication using Content-Defined Chunking (CDC) with near-duplicate detection using SimHash and Faiss. The performance-critical core is implemented in C++ and parallelized with OpenMP, offering a significant speedup over pure Python implementations.

This tool is ideal for cleaning large text datasets for **training large language models**, data analysis, or any application requiring the removal of both exact and near-duplicate content. It is very efficient, requires about 480s for a 1-GB text dataset on my desktop, and almost scales linearly with the data size and the CPU cores.

## Features

- **Multi-Stage Deduplication Pipeline**:
  1. **Exact Substring Deduplication (CDC)**: A fast, parallelized Content-Defined Chunking stage removes redundant text blocks across the entire dataset, significantly reducing data volume.
  2. **Near-Duplicate Document Detection**: A SimHash and Faiss-powered stage identifies and removes documents that are nearly identical (e.g., minor edits, different timestamps).
- **High-Performance C++ Core**: The core logic is written in C++ and leverages modern libraries for maximum performance:
  - **Parallel Processing**: Uses OpenMP to utilize all available CPU cores.
  - **Abseil**: Employs Google's high-performance `flat_hash_map` and `flat_hash_set`.
  - **Faiss**: Integrates Facebook AI's library for efficient similarity search on binary vectors.
  - **AVX2 Vectorization**: Uses SIMD instructions to accelerate SimHash signature generation.
- **Robust and Safe**: Handles Unicode encoding issues and is memory-safe.
- **Seamless Python Integration**: Exposes a clean Python API via pybind11, designed to integrate smoothly with libraries like Hugging Face `datasets`.

## Requirements

### 1. System Dependencies
- A modern C++ compiler that supports C++17 (e.g., GCC 9+).
- CMake (version 3.12+).
- OpenMP for parallelization.

On CentOS/RHEL, you can install these with:
```bash
# Example for installing GCC Toolset 11 on CentOS 8
sudo dnf install gcc-toolset-11 cmake

# Activate the new compiler for your session
scl enable gcc-toolset-11 bash
```
On Ubuntu/Debian, you can install these with:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libomp-dev
```
2. Faiss
You need to have Faiss installed on your system. It's recommended to install it with conda, especially with AVX2 support.
Please follow the official Faiss installation guide. Ensure you build and install the C++ library. If you want to build from source, a typical installation might look like this:
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build .
make -C build -j$(nproc)
sudo make -C build install
```
3. Python Dependencies
The project requires several Python packages. You can install them using pip:
```bash
pip install -r requirements.txt
```
Installation
This project uses scikit-build-core to compile the C++ extension. Once all system and Python dependencies are met, you can install the toolkit directly using pip from the project's root directory.
```bash
# Clone this repository
git clone https://github.com/conanhujinming/text_dedup.git
cd text_dedup

# Recommended: Add Abseil as a git submodule
git submodule add https://github.com/abseil/abseil-cpp.git third_party/abseil-cpp
git submodule update --init --recursive

# Install the package. This will compile the C++ core.
pip install .
```
The installation command will automatically invoke CMake to build the C++ extension and place it correctly within your Python environment.
## Usage
The toolkit is designed to work seamlessly with Hugging Face datasets. Here is a complete example of how to use it.
```Python
import dedup_cpp_core  # This is the compiled C++ module
from datasets import Dataset
from tqdm import tqdm

# 1. Load your dataset
# For demonstration, we create a sample dataset
data = {
    'text': [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy canine.", # Near-duplicate
        "This is a completely unique sentence.",
        "The quick brown fox jumps over the lazy dog.", # Exact duplicate
        "Another unique piece of text."
    ],
    'id': [1, 2, 3, 4, 5]
}
dataset = Dataset.from_dict(data)
print("Original Dataset:")
print(dataset.to_pandas())

# 2. Prepare the data
# Clean the text and extract it into a list for the C++ function
print("\nCleaning and preparing data...")
docs_to_process = [doc for doc in tqdm(dataset['text'])]

# 3. Run the deduplication
# The C++ function returns a list of the same size as the input.
# Duplicates are replaced with `None`.
print("\nStarting C++ deduplication...")
updated_docs_or_none = dedup_cpp_core.deduplicate_cpp(
    docs=docs_to_process,
    min_length_dedup=64,
    hamming_threshold=3,
    faiss_index_type="hash",  # Recommended for this use case. "IVF" and "flat" also available.
    simhash_bits=64
)
print("C++ deduplication finished.")

# 4. Integrate results back into the dataset
# Add the result as a new column
dataset = dataset.add_column("updated_text", updated_docs_or_none)

# Filter out the rows where the new text is None (i.e., duplicates)
final_dataset = dataset.filter(lambda example: example["updated_text"] is not None)

# Clean up the columns
final_dataset = final_dataset.remove_columns(["text"])
final_dataset = final_dataset.rename_column("updated_text", "text")

# 5. View the result
print("\nFinal Deduplicated Dataset:")
print(final_dataset.to_pandas())
```
## Configuration Parameters
The deduplicate_cpp function takes several important parameters:

docs (List[str]): A list of documents to deduplicate.

min_length_dedup (int): Controls the Content-Defined Chunking. It acts as both the minimum chunk size and the divisor for the rolling hash, influencing the average chunk size. A smaller value (e.g., 64, 128) is better for fine-grained deduplication.

hamming_threshold (int): The maximum Hamming distance for two documents to be considered near-duplicates. For a 64-bit SimHash, a value between 3 and 7 is typical.

faiss_index_type (str): The type of Faiss index to use.

"hash" (Recommended Default): Fast to build and query, good balance for this task.

"IVF": Slower to build but very fast to query. Better for scenarios where you build an index once and query it many times.

"flat": Brute-force search. Guarantees 100% accuracy but is slow for large datasets. Useful for debugging or small-scale tasks.

simhash_bits (int, optional, default: 64): The number of bits in the SimHash signature. Must be a multiple of 8.
## How It Works
The deduplication process is a pipeline:

### Content-Defined Chunking (CDC)

Each document is broken down into chunks based on the content itself using a rolling hash function. This ensures that if a block of text is inserted or deleted, only the chunks around that change are affected, while other identical blocks across different documents will still produce the same chunks and hashes. A global set of seen chunk hashes is maintained to discard any duplicate chunk after its first appearance.

### Text Reconstruction

After discarding duplicate chunks, the remaining unique chunks for each document are concatenated to form a cleaned version of the text.

### SimHash Generation

Each cleaned document is featurized (tokenized into words), and a SimHash signature (a compact binary fingerprint) is generated. Documents that are semantically similar will have SimHash signatures with a small Hamming distance between them.

### Faiss Indexing and Search

All SimHash signatures are added to a Faiss index. A range_search is then performed to efficiently find all pairs of documents whose signatures are within the specified hamming_threshold.

### Clustering and Filtering

A Union-Find data structure is used to group documents into clusters of near-duplicates based on the Faiss search results. For each cluster, only one document (the one with the lowest original index) is kept, and the rest are marked for removal.

## Contribution
Feel free to open issues or submit pull requests. Any contributions to improve performance, accuracy, or usability are welcome.