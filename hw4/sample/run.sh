srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 00:04:00 \
    time \
        ./hw4 \
            ../testcases/case00.in \
            case00.out