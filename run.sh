#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

declare -a DATASETS=(
  "email-Enron.txt https://snap.stanford.edu/data/email-Enron.txt.gz"
  "wiki-Vote.txt https://snap.stanford.edu/data/wiki-Vote.txt.gz"
  "as-skitter.txt https://snap.stanford.edu/data/as-skitter.txt.gz"
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
g++ -O3 -march=native -std=c++17 -fopenmp betweenness.cpp -o betweenness

: > results.txt

for entry in "${DATASETS[@]}"; do
  filename="${entry%% *}"
  # derive per-dataset output filename: email-Enron.txt -> result_email-Enron.txt
  result_file="result_${filename}"

  echo "Running dataset: $filename"

  # run, save to individual file, and also append to combined results.txt
  ./betweenness "$filename" | tee "$result_file" >> results.txt
  echo "" >> results.txt

  echo "  -> saved to $result_file"
done

write_report