#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

declare -a DATASETS=(
  "email-Enron.txt https://snap.stanford.edu/data/email-Enron.txt.gz"
  "as-skitter.txt https://snap.stanford.edu/data/as-skitter.txt.gz"
  "wiki-Vote.txt https://snap.stanford.edu/data/wiki-Vote.txt.gz"
)

write_report() {
  {
    echo "A. Team Information"
    echo "[PLACEHOLDER - fill in your team's names and IDs]"
    echo "Student 1: Name -- ID"
    echo "Student 2: Name -- ID"
    echo
    echo "B. Individual Contributions"
    echo "[PLACEHOLDER - fill in each member's contribution]"
    echo
    echo "C. Program Output"
    echo
    if [[ -f results.txt ]]; then
      cat results.txt
    else
      echo "[Run \`bash run.sh\` to populate this section.]"
    fi
  } > report.txt
}

download_and_prepare() {
  local filename="$1"
  local url="$2"
  local gz_file="${filename}.gz"

  if [[ -f "$filename" ]]; then
    echo "Using existing dataset: $filename"
    return
  fi

  if [[ ! -f "$gz_file" ]]; then
    echo "Downloading $filename"
    wget -O "$gz_file" "$url"
  else
    echo "Using existing archive: $gz_file"
  fi

  echo "Decompressing $gz_file"
  gunzip -c "$gz_file" > "$filename"
}

write_report

for entry in "${DATASETS[@]}"; do
  filename="${entry%% *}"
  url="${entry#* }"
  download_and_prepare "$filename" "$url"
done

echo "Compiling betweenness.cpp"
# -fopenmp enables parallelism; -O3 + -march=native give extra ~20% on top
g++ -O3 -march=native -std=c++17 -fopenmp betweenness.cpp -o betweenness

: > results.txt

for entry in "${DATASETS[@]}"; do
  filename="${entry%% *}"
  {
    echo "Running dataset: $filename"
    ./betweenness "$filename"
    echo
  } | tee -a results.txt
done

write_report
