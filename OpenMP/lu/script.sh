#!/bin/bash

# Dynamically resolve the target file names based on the directory name,
# replicating the $(shell basename `pwd`) behavior defined in the Makefile.
DIR_NAME=$(basename "$PWD")
TARGET_C="${DIR_NAME}.c"
EXEC_NAME="./${DIR_NAME}_acc"

# Define the separate source files for the baseline and optimized versions.
FILE_BASELINE="lu_baseline.c"
FILE_OPTIMIZED="lu_optimized.c"

# Parse command-line arguments to determine which datasets to evaluate.
# If no arguments are provided, default to the standard PolyBench dataset suite.
if [ "$#" -eq 0 ]; then
    DATASETS=("MINI" "SMALL" "STANDARD" "LARGE" "EXTRALARGE")
else
    DATASETS=("$@")
fi

# Initialize the directory structure for performance logs.
mkdir -p results/baseline/perf results/baseline/gprof results/baseline/valgrind results/optimized/perf results/optimized/valgrind env

# Helper function to construct the appropriate C preprocessor flags for perf.
get_dataset_flags() {
    if [ "$1" == "STANDARD" ]; then
      echo "-DPOLYBENCH_TIME"
    else
      echo "-D${1}_DATASET -DPOLYBENCH_TIME"
    fi
}

echo ""
echo "Checking 'perf' profiler..."

if ! command -v perf &> /dev/null; then
    echo "[WARNING] 'perf' profiler not detected. Attempting to load module perf/1.0..."
    module load perf/1.0
    echo "[DONE] Module perf/1.0 loaded successfully."
fi

echo ""
echo "Saving environment..."

{
  echo "=== System & Board Info ==="
  uname -a
  
  gcc --version | head -n 1
  
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}"
    

} > env/environment_conditions.txt

echo "[SUCCESS] Environment conditions saved successfully!"

# Defining PERF_EVENTS variable
PERF_EVENTS="cycles:u,instructions:u,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses"

echo ""
echo "========================================"
echo "     PROFILING BASELINE EXECUTION"
echo "========================================"
echo ""

cp $FILE_BASELINE $TARGET_C

for DS in "${DATASETS[@]}"; do

    FLAGS=$(get_dataset_flags $DS)
    
    echo " -> Evaluating $DS dataset..."
    
    if [ "$DS" == "SMALL" ]; then
      
      # Compiling WITHOUT DUMP_ARRAYS and WITHOUT POLYBENCH_TIME for Valgrind
      echo "Running Valgrind..."
      
      make OPT="-O2 -g" EXT_CFLAGS="-DSMALL_DATASET" clean all &> /dev/null
      
      valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=results/baseline/valgrind/cachegrind.out $EXEC_NAME &> results/baseline/valgrind/SMALL_cachegrind.txt
      cg_annotate --auto=yes results/baseline/valgrind/cachegrind.out > results/baseline/valgrind/SMALL_cg_annotate.txt
      
      echo "[SUCCESS] Cachegrind telemetry saved successfully!"
      
    elif [ "$DS" == "STANDARD" ]; then
    
      # Compiling with DUMP_ARRAYS only for correctness check
      make OPT="-O2 -g" EXT_CFLAGS="$FLAGS -DPOLYBENCH_DUMP_ARRAYS" clean all &> /dev/null
      
      $EXEC_NAME 2> results/baseline/baseline_dump.txt > /dev/null
    
      # Compiling WITHOUT POLYBENCH_TIME for Gprof
      echo "Running Gprof..."
      
      make OPT="-O0 -pg" EXT_CFLAGS="" clean all &> /dev/null
      
      $EXEC_NAME &> /dev/null
      
      gprof $EXEC_NAME gmon.out > results/baseline/gprof/baseline_standard_gprof_flat.txt
      
      echo "[SUCCESS] GPROF flat profile saved successfully!"
      
      rm -f gmon.out
    
    fi
    
    # Compiling with POLYBENCH_TIME for native perf runs
    make OPT="-O2 -g" EXT_CFLAGS="$FLAGS" clean all &> /dev/null
    
    # Warm-up run
    echo "Executing warm-up run..."
    
    $EXEC_NAME &> /dev/null
    
    echo "Warm-up run completed. Executing 'perf' command..."
  
    perf stat -r 10 -e "$PERF_EVENTS" $EXEC_NAME &> "results/baseline/perf/${DS}.txt"

    echo "Saving temperature metric..."
    echo "[BASELINE - $DS] | CPU Temp: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000))˚C" >> env/environment_conditions.txt
    echo "[COMPLETED] Temperature metric logged successfully!"

    echo "[COMPLETED] Performance metrics logged to results/baseline/perf/${DS}.txt!"
    echo ""
    
done

echo ""
echo "========================================"
echo "     PROFILING OPTIMIZED EXECUTION"
echo "========================================"
echo ""

cp $FILE_OPTIMIZED $TARGET_C

for DS in "${DATASETS[@]}"; do

    FLAGS=$(get_dataset_flags $DS)
    
    echo " -> Evaluating $DS dataset..."
    
    if [ "$DS" == "SMALL" ]; then
      
      # Compiling WITHOUT DUMP_ARRAYS and WITHOUT POLYBENCH_TIME for Valgrind
      echo "Running Valgrind..."
      
      make OPT="-O2 -g -fopenmp" EXT_CFLAGS="-DSMALL_DATASET" clean all &> /dev/null
      
      valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=results/optimized/valgrind/cachegrind.out $EXEC_NAME &> results/optimized/valgrind/SMALL_cachegrind.txt
      cg_annotate --auto=yes results/optimized/valgrind/cachegrind.out > results/optimized/valgrind/SMALL_cg_annotate.txt
      
      echo "[COMPLETED] Cachegrind telemetry saved successfully!"
      
    elif [ "$DS" == "STANDARD" ]; then
    
      # Compiling with DUMP_ARRAYS only to verify correctness against baseline
      make OPT="-O2 -g -fopenmp" EXT_CFLAGS="$FLAGS -DPOLYBENCH_DUMP_ARRAYS" clean all &> /dev/null
      
      $EXEC_NAME 2> results/optimized/optimized_dump.txt > /dev/null    
      
      if diff -q results/baseline/baseline_dump.txt results/optimized/optimized_dump.txt > /dev/null; then
      
        echo " [OK] Correctness verified: Baseline and Optimized outputs match!"
        
      else
      
        echo " [WARNING] Baseline and Optimized outputs differ!"
        
      fi
      
      rm -f results/baseline/baseline_dump.txt results/optimized/optimized_dump.txt
      
    fi
    
    # Compiling with POLYBENCH_TIME for native perf runs
    make OPT="-O2 -g -fopenmp" EXT_CFLAGS="$FLAGS" clean all &> /dev/null
    
    # Warm-up run
    echo "Executing warm-up run..."
    
    $EXEC_NAME &> /dev/null
    
    echo "Warm-up run completed. Executing 'perf' command..."
    
    perf stat -r 10 -e "$PERF_EVENTS" $EXEC_NAME &> "results/optimized/perf/${DS}.txt"
    
    echo "Saving temperature metric..."
    echo "[OPTIMIZED - $DS] | CPU Temp: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000))˚C" >> env/environment_conditions.txt
    echo "[COMPLETED] Temperature metric saved successfully!"

    echo "[COMPLETED] Performance metrics logged to results/optimized/perf/${DS}.txt!"
    echo ""
    
done

echo ""
echo "Benchmarking procedure completed. Review the 'results' directory for detailed telemetry."
echo ""
