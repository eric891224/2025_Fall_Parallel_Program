# Test configuration
TID=04
TIME_LIMIT=00:03:00
ACCOUNT=ACD114118

# Create results directory for analysis
mkdir -p ./analysis_results

# ===================================
# 1. Scaling Number of Processes (Fixed: 1 node, 6 cores/process)
# ===================================
echo -e "\n[Test 1] Scaling Number of Processes"
for proc in 1 2 4 6 8; do
    echo "Testing with $proc processes..."
    srun -A $ACCOUNT -N 1 -n $proc -c 6 --time $TIME_LIMIT \
        ./hw2 \
        ./testcases/$TID.jpg \
        ./analysis_results/${TID}_proc${proc}.jpg \
        ./analysis_results/${TID}_proc${proc}.txt
done