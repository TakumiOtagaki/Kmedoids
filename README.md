# K-medoids

A Python script for K-medoids clustering with multi-process.

# Introduction
There is no useful multiprocessing implementation of K-medoids.
This program can be used when you have distance_matrix (triangle is OK).

# usage
```
python kmedoids-parallel.py --help
usage: python kmedoids-parallel.py [-h] [--num_thread NUM_THREAD]
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
  -c, --av_cpu          Check available CPU. If True, the program will exit after checking available CPU
  -p NUM_THREAD, --num_thread NUM_THREAD
                        Number of threads. if num_thread > num_points, set num_thread = num_points for avoiding useless cpu usage
  -s INPUT_SEP, --input_sep INPUT_SEP
                        Input distance matrix separator
  -I INPUT_DISTMAT, --input_distmat INPUT_DISTMAT
                        Input distance matrix
  -T DIST_TYPE, --dist_type DIST_TYPE
                        Input distance matrix type (triu|tril|sym)
  -M OUTPUT_MEDOIDS, --output_medoids OUTPUT_MEDOIDS
                        Output medoids
  -L OUTPUT_LABEL, --output_label OUTPUT_LABEL
                        Output labels
  -k NUM_CLUSTERS, --num_clusters NUM_CLUSTERS
                        Number of clusters
  -v, --verbose         Verbose mode
  -N MAX_ITER, --max_iter MAX_ITER
                        Maximum number of iterations
  -r RANDOM_SEED, --random_seed RANDOM_SEED
                        Random seed.
```

For example, 
```for_example.sh
python kmedoids.py --num_core 4 --input_distmat dist.csv --dist_type (triu|tril|sym) \
                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000 \
                    --verbose --random_seed 0

python ../../kmedoids-parallel/kmedoids-parallel.py --num_thread 30 --input_distmat edit_dist_zerofilled.csv --dist_type triu --output_medoids kmedoids_result/medois.csv --output_label kmedoids_result/labels.csv --num_clusters 30 --max_iter 1000 --verbose
```


# Installation
Easy.
Python3 will work.

- Python 3.11.6


```installation.sh
$ git clone https://github.com/TakumiOtagaki/Kmedoids-parallel.git
$ cd Kmedoids-parallel
$ pip install -r requirements.txt
```

If you want, you can prepend `kmedoids.py` to $PATH.
