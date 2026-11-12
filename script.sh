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

# Measures the CPU usage of OTHER processes: samples /proc/stat for 1 second
# while this script runs nothing, and prints the busy share of all cores (%).
# Unlike the load average, it is not inflated by our own previous runs.
measure_background_cpu() {

  local c1 c2 i total=0 idle
  
  read -r -a c1 <<< "$(head -n 1 /proc/stat)"
  
  sleep 1
  
  read -r -a c2 <<< "$(head -n 1 /proc/stat)"
  
  # Fields: cpu user nice system idle iowait irq softirq steal
  idle=$(( (c2[4] + c2[5]) - (c1[4] + c1[5]) ))
  
  for i in 1 2 3 4 5 6 7 8; do
  
    total=$(( total + c2[i] - c1[i] ))
      
  done
  
  echo $(( total > 0 ? 100 * (total - idle) / total : 0 ))
  
}

echo ""
echo "Checking 'perf' profiler..."

if ! command -v perf &> /dev/null; then

    echo "[WARNING] 'perf' profiler not detected. Attempting to load module perf/1.0. Please wait..."
    
    module load perf/1.0
    
    echo "[DONE] Module perf/1.0 loaded successfully."
    
fi

echo ""
echo "Checking 'clang' and 'cuda' profiler..."

if ! command -v clang &> /dev/null; then

    echo "[WARNING] 'clang' profiler not detected. Attempting to load modules clang/11.0.0 and cuda/10.0. Please wait..."
    
    module load clang/11.0.0 cuda/10.0
    
    echo "[DONE] Modules clang/11.0.0 and cuda/10.0 loaded successfully."

fi

echo ""
echo "Saving environment..."

{
  echo "=== System & Board Info ==="
  uname -a
  
  gcc --version | head -n 1
  
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}"
  echo "OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-default}"

  echo "=== CPU State ==="
  echo "Online CPUs: $(cat /sys/devices/system/cpu/online)"
  echo "Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "Max freq (kHz): $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo n/a)"
  
  # Jetson power mode (MAXN = 4 cores at full clock, 5W = 2 cores at lower clock)
  if command -v nvpmodel &> /dev/null; then
  
    nvpmodel -q 2>/dev/null | grep -i "power mode" || echo "Power mode: unknown"
    
  else
  
    echo "Power mode: n/a"
    
  fi

  # Other activity on the machine: background load makes the OpenMP barriers wait
  echo "=== System Load at Start ==="
  
  uptime

  echo "=== Per-dataset Conditions (Background CPU = other processes, % of all cores, sampled for 1 s before the run) ==="

} > env/environment_conditions.txt

echo "[SUCCESS] Environment conditions saved successfully!"

# Defining PERF_EVENTS variable.
# On ARM (Cortex-A57) the generic cache-references/cache-misses events count the
# same L1D events as L1-dcache-*, so on aarch64 they are replaced with raw ARMv8
# PMU events: r16 = L2D_CACHE (L2 accesses), r17 = L2D_CACHE_REFILL (L2 misses).
if [ "$(uname -m)" == "aarch64" ]; then

  PERF_EVENTS="cycles:u,instructions:u,L1-dcache-loads,L1-dcache-load-misses,r16,r17"
  
else

  PERF_EVENTS="cycles:u,instructions:u,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses"
  
fi

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
      echo " Running Valgrind..."
      
      make OPT="-O2 -g" EXT_CFLAGS="-DSMALL_DATASET" clean all &> /dev/null
      
      valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=results/baseline/valgrind/cachegrind.out $EXEC_NAME &> results/baseline/valgrind/SMALL_cachegrind.txt
      cg_annotate --auto=yes results/baseline/valgrind/cachegrind.out > results/baseline/valgrind/SMALL_cg_annotate.txt
      
      echo " [SUCCESS] Cachegrind telemetry saved successfully!"
      
    elif [ "$DS" == "STANDARD" ]; then
    
      # Compiling with DUMP_ARRAYS only for correctness check
      make OPT="-O2 -g -ffp-contract=off" EXT_CFLAGS="$FLAGS -DPOLYBENCH_DUMP_ARRAYS" clean all &> /dev/null
      
      $EXEC_NAME 2> results/baseline/baseline_dump.txt > /dev/null
    
      # Compiling WITHOUT POLYBENCH_TIME for Gprof
      echo " Running Gprof..."
      
      make OPT="-O0 -pg" EXT_CFLAGS="" clean all &> /dev/null
      
      $EXEC_NAME &> /dev/null
      
      gprof $EXEC_NAME gmon.out > results/baseline/gprof/baseline_standard_gprof_flat.txt
      
      echo " [SUCCESS] GPROF flat profile saved successfully!"
      
      rm -f gmon.out
    
    fi
    
    # Compiling with POLYBENCH_TIME for native perf runs
    make OPT="-O2 -g" EXT_CFLAGS="$FLAGS" clean all &> /dev/null
    
    # Record background CPU usage before running (other activity skews timings)
    BG_CPU=$(measure_background_cpu)
    
    # Warm-up run
    echo " Executing warm-up run..."
    
    $EXEC_NAME &> /dev/null
    
    echo " Warm-up run completed. Executing 'perf' command..."
  
    perf stat -r 10 -e "$PERF_EVENTS" $EXEC_NAME &> "results/baseline/perf/${DS}.txt"

    echo " Saving temperature metric..."
    echo " [BASELINE - $DS] | CPU Temp: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000))˚C" >> env/environment_conditions.txt
    echo " [COMPLETED] Temperature metric logged successfully!"

    echo " [COMPLETED] Performance metrics logged to results/baseline/perf/${DS}.txt!"
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
      echo " Running Valgrind..."
      
      make OPT="-O2 -g -fopenmp" EXT_CFLAGS="-DSMALL_DATASET" clean all &> /dev/null
      
      valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=results/optimized/valgrind/cachegrind.out $EXEC_NAME &> results/optimized/valgrind/SMALL_cachegrind.txt
      cg_annotate --auto=yes results/optimized/valgrind/cachegrind.out > results/optimized/valgrind/SMALL_cg_annotate.txt
      
      echo " [COMPLETED] Cachegrind telemetry saved successfully!"
      
    fi
    
    # Compiling with POLYBENCH_TIME for native perf runs
    # STANDARD, LARGE, and EXTRALARGE go to the GPU using Clang.
    if [ "$DS" == "STANDARD" ] || [ "$DS" == "LARGE" ] || [ "$DS" == "EXTRALARGE" ]; then
    
      if [ "$DS" == "STANDARD" ]; then
      
        # Compiling with DUMP_ARRAYS only to verify correctness against baseline
        clang -O2 -g -ffp-contract=off -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -I. $FLAGS -DPOLYBENCH_DUMP_ARRAYS -I../../../utilities $TARGET_C ../../../utilities/polybench.c -o ${DIR_NAME}_acc -lm &> /dev/null
        
        $EXEC_NAME 2> results/optimized/optimized_dump.txt > /dev/null    
        
        if diff -q results/baseline/baseline_dump.txt results/optimized/optimized_dump.txt > /dev/null; then
        
          echo " [OK] Correctness verified: Baseline and Optimized outputs match!"
          
        else
        
          echo " [WARNING] Baseline and Optimized outputs differ!"
          
        fi
        
        rm -f results/baseline/baseline_dump.txt results/optimized/optimized_dump.txt
      
      fi
      
      clang -O2 -g -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -I. $FLAGS -I../../../utilities $TARGET_C ../../../utilities/polybench.c -o ${DIR_NAME}_acc -lm &> /dev/null
      
    else
    
      make OPT="-O2 -g -fopenmp" EXT_CFLAGS="$FLAGS" clean all &> /dev/null
      
    fi
    
    # Record background CPU usage before running (other activity skews timings)
    BG_CPU=$(measure_background_cpu)
    
    # Warm-up run
    echo " Executing warm-up run..."
    
    $EXEC_NAME &> /dev/null
    
    echo " Warm-up run completed. Executing 'perf' command..."
    
    perf stat -r 10 -e "$PERF_EVENTS" $EXEC_NAME &> "results/optimized/perf/${DS}.txt"
    
    echo " Saving temperature metric..."
    echo " [OPTIMIZED - $DS] | CPU Temp: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000))˚C" >> env/environment_conditions.txt
    echo " [COMPLETED] Temperature metric saved successfully!"

    echo " [COMPLETED] Performance metrics logged to results/optimized/perf/${DS}.txt!"
    echo ""
    
done

echo ""
echo "Benchmarking procedure completed. Review the 'results' directory for detailed telemetry."
echo ""