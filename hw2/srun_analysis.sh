#!/bin/bash
#SBATCH --job-name=hw2_analysis
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=1                    # Request max nodes you need
#SBATCH --ntasks=4                  # Request max tasks you need
#SBATCH --cpus-per-task=8           # Request max CPUs per task
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Test configuration
TID=04
TIME_LIMIT=00:10:00
ACCOUNT=ACD114118

# Create results directory for analysis
mkdir -p ./analysis_results

echo "==================================="
echo "Scalability Analysis for HW2"
echo "==================================="

# # ===================================
# # 1. Scaling Number of Processes (Fixed: 1 node, 6 cores/process)
# # ===================================
# echo -e "\n[Test 1] Scaling Number of Processes"
# for proc in 1 2 4 6 8; do
#     echo "Testing with $proc processes..."
#     srun -A $ACCOUNT -N 1 -n $proc -c 6 --time $TIME_LIMIT \
#         ./hw2 \
#         ./testcases/$TID.jpg \
#         ./analysis_results/${TID}_proc${proc}.jpg \
#         ./analysis_results/${TID}_proc${proc}.txt
# done

# ===================================
# 2. Scaling Number of Cores per Process (Fixed: 1 node, 4 processes)
# ===================================
echo -e "\n[Test 2] Scaling Number of Cores per Process"
for core in 1 2 4 6 8; do
    echo "Testing with $core cores per process..."
    srun -A $ACCOUNT -N 1 -n 4 -c $core --time $TIME_LIMIT \
        ./hw2 \
        ./testcases/$TID.jpg \
        ./analysis_results/${TID}_core${core}.jpg \
        ./analysis_results/${TID}_core${core}.txt
done

# # ===================================
# # 3. Scaling Number of Nodes (Fixed: 4 processes/node, 6 cores/process)
# # ===================================
# echo -e "\n[Test 3] Scaling Number of Nodes"
# for node in 1 2 4 6 8; do
#     # total_proc=$((node * 4))
#     total_proc=$node
#     echo "Testing with $node nodes ($total_proc total processes)..."
#     srun -A $ACCOUNT -N $node -n $total_proc -c 6 --time $TIME_LIMIT \
#         ./hw2 \
#         ./testcases/$TID.jpg \
#         ./analysis_results/${TID}_node${node}.jpg \
#         ./analysis_results/${TID}_node${node}.txt
# done

# # ===================================
# # 4. Strong Scaling (Fixed total work, increase resources)
# # ===================================
# echo -e "\n[Test 4] Strong Scaling - Fixed Problem Size"
# # Keep total cores constant, vary process/thread distribution
# configs=(
#     "1 1 24"   # 1 node, 1 process, 24 cores
#     "1 2 12"   # 1 node, 2 processes, 12 cores each
#     "1 4 6"    # 1 node, 4 processes, 6 cores each
#     "1 8 3"    # 1 node, 8 processes, 3 cores each
#     "1 12 2"   # 1 node, 12 processes, 2 cores each
#     "2 12 2"   # 2 nodes, 12 processes, 2 cores each
# )

# for config in "${configs[@]}"; do
#     read -r n p c <<< "$config"
#     echo "Testing N=$n, processes=$p, cores/proc=$c..."
#     srun -A $ACCOUNT -N $n -n $p -c $c --time $TIME_LIMIT \
#         ./hw2 \
#         ./testcases/$TID.jpg \
#         ./analysis_results/${TID}_strong_n${n}_p${p}_c${c}.jpg \
#         ./analysis_results/${TID}_strong_n${n}_p${p}_c${c}.txt
# done

# # ===================================
# # 5. Weak Scaling (Increase problem size with resources)
# # ===================================
# echo -e "\n[Test 5] Weak Scaling - Scale Problem and Resources Together"
# # Use different test cases if available, or same test case with different process counts
# test_cases=(01 02 04)  # Replace with larger test cases if available
# process_counts=(2 4 8)

# for i in ${!process_counts[@]}; do
#     proc=${process_counts[$i]}
#     test=${test_cases[$i]}
#     echo "Testing weak scaling: $proc processes with test case $test..."
#     srun -A $ACCOUNT -N 1 -n $proc -c 6 --time $TIME_LIMIT \
#         ./hw2 \
#         ./testcases/${test}.jpg \
#         ./analysis_results/${test}_weak_proc${proc}.jpg \
#         ./analysis_results/${test}_weak_proc${proc}.txt
# done

# # ===================================
# # 6. Hybrid Configuration Analysis
# # ===================================
# echo -e "\n[Test 6] Hybrid MPI+OpenMP Configurations"
# configs=(
#     "1 1 24"   # Pure threading
#     "1 6 4"    # Balanced hybrid
#     "1 24 1"   # Pure MPI
#     "2 12 2"   # 2 nodes, hybrid
# )

# for config in "${configs[@]}"; do
#     read -r n p c <<< "$config"
#     echo "Testing hybrid: N=$n, MPI ranks=$p, threads/rank=$c..."
#     srun -A $ACCOUNT -N $n -n $p -c $c --time $TIME_LIMIT \
#         ./hw2 \
#         ./testcases/$TID.jpg \
#         ./analysis_results/${TID}_hybrid_n${n}_p${p}_c${c}.jpg \
#         ./analysis_results/${TID}_hybrid_n${n}_p${p}_c${c}.txt
# done

echo -e "\n==================================="
echo "All tests completed!"
echo "Results saved in ./analysis_results/"
echo "==================================="