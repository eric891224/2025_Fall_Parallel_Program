node=1
proc=4
core=6

TID=01

srun -A ACD114118 -N $node -n $proc -c $core --time 00:03:00 \
    ./hw2 \
    ./testcases/$TID.jpg \
    ./results/$TID.jpg \
    ./results/$TID.txt