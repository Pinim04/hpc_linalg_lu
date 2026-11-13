#!/bin/bash
# Measures the two roofs of the roofline model on the Jetson Nano:
#   - memory roof: STREAM (Copy/Triad) for 1..4 threads
#   - compute roof: peak_flops.c (NEON FP64 FMA) for 1..4 threads
# Usage (from this folder, on the Jetson):  bash run_roofline.sh > roofline.txt 2>&1

cd "$(dirname "$0")" || exit 1

echo "===== HARDWARE ====="
lscpu
echo "cpuinfo_max_freq (kHz): $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)"
echo "scaling_cur_freq (kHz): $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tr '\n' ' ')"
echo "nvpmodel: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')"

echo "===== CLOCKS (CPU + EMC/DRAM) ====="
sudo jetson_clocks --show 2>&1 || echo "[jetson_clocks --show not available]"
sudo cat /sys/kernel/debug/clk/emc/clk_rate 2>/dev/null | sed 's/^/emc clk_rate (Hz): /'

echo "===== BUILD ====="
if [ ! -f stream.c ]; then
  wget -q https://www.cs.virginia.edu/stream/FTP/Code/stream.c || { echo "[ERROR] could not download stream.c"; }
fi
gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=20000000 -DNTIMES=20 stream.c -o stream
gcc -O2 -fopenmp peak_flops.c -o peak_flops

echo "CPU temp start: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000)) C"

for t in 1 2 3 4; do
  echo "===== THREADS=$t ====="
  OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./stream | grep -E 'Copy|Scale|Add|Triad'
  OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./peak_flops
done

echo "CPU temp end: $(($(cat /sys/class/thermal/thermal_zone1/temp)/1000)) C"
