# K-medoids

A Python script for K-medoids clustering with multi-process.

# Introduction
There is no useful multiprocessing implementation of K-medoids.

```.sh
python Kmedoids.py --num_thread 64 --input_distmat dist.csv --dist_type (triu|tril|sym) \
 --output_medoids out_medoids.csv --output_label out_labels.csv
```

# usage
```
usage: kmedoids-parallel.py [-h] [--num_thread NUM_THREAD]
                            [--input_distmat INPUT_DISTMAT]
                            [--dist_type DIST_TYPE]
                            [--output_medoids OUTPUT_MEDOIDS]
                            [--output_label OUTPUT_LABEL]
                            [--num_clusters NUM_CLUSTERS] [--verbose]
                            [--max_iter MAX_ITER] [--av_cpu]
                            [--random_seed RANDOM_SEED]

Kmedoids clustering

options:
  -h, --help            show this help message and exit
  --num_thread NUM_THREAD
                        Number of threads. if num_thread > num_points, set num_thread = num_points for avoiding useless cpu usage
  --input_distmat INPUT_DISTMAT
                        Input distance matrix
  --dist_type DIST_TYPE
                        Input distance matrix type (triu|tril|sym)
  --output_medoids OUTPUT_MEDOIDS
                        Output medoids
  --output_label OUTPUT_LABEL
                        Output labels
  --num_clusters NUM_CLUSTERS
                        Number of clusters
  --verbose             Verbose mode
  --max_iter MAX_ITER   Maximum number of iterations
  --av_cpu              Check available CPU. If True, the program will exit after checking available CPU
  --random_seed RANDOM_SEED
                        Random seed.
```

For example, 
```for_example.sh
python Kmedoids.py --num_core 4 --input_distmat dist.csv --dist_type (triu|tril|sym) \
                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000 \
                    --verbose --random_seed 0
```