# K-medoids

A Python script for K-medoids clustering with multi-process.

# Introduction
There is no useful multiprocessing implementation of K-medoids.
This program can be used when you have distance_matrix (triangle is OK).

# usage
```
$ python kmedoids-parallel.py --help
usage: python kmedoids-parallel.py [-h] [-c] [-p NUM_THREAD] [-s INPUT_SEP] [-I INPUT_DISTMAT] [-T DIST_TYPE]
                            [-M OUTPUT_MEDOIDS] [-L OUTPUT_LABEL] [-k NUM_CLUSTERS] [-v] [-N MAX_ITER]
                            [-r RANDOM_SEED]
Kmedoids clustering with multiprocessing.

options:
  -h, --help            show this help message and exit
  -c, --av_cpu          Checking the available CPU. If True, the program will exit after checking available CPU
  -p NUM_THREAD, --num_thread NUM_THREAD
                        Number of threads. if num_thread > num_points, set num_thread = num_points for avoiding
                        useless cpu usage
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
                        Random seed: Should be integer
```

For example, 
```for_example.sh
python kmedoids.py --num_core 4 --input_distmat dist.csv --dist_type (triu|tril|sym) \
                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000 \
                    --verbose --random_seed 0

python kmedoids-parallel.py \
 -I test/distmat.N100.sym.csv \
 -M test/medoids.N100.sym.csv \
 -L test/labels.N100.sym.csv \
 -N 100 \
 -k 5 \
 -p 10
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

```~/.bashrc
export PATH="path/to/kmedoids-parallel.py:${PATH}"

```

# example
```
$ cd Kmedoids-parallel/test
$ python kmedoids-parallel.py -I 'test/test_distmat_n300.csv' -s , -T sym -M test/test300.medoid.csv -L test/test300.label.csv -k 4 -v -N 10 -r 0 -p 2
Reading distance matrix...
Reading distmat Done
        Distance matrix shape: (300, 300)
        Reading distmat: Time elapsed =  0.01831841468811035 s
--------Input confirmation--------
k=4, num_thread=1, max_iter=10, random_seed=0
-------Clustering Starts...-------
kmedoids normal initialization.
...Initialization done.
         medoids = [197, 215, 20, 132]
         labels.freq = [50, 56, 121, 73] 
Main loop starts....
Iteration 0: 0.0 %
         medoids = [250 254  25 163]
         labels.freq = [59, 47, 121, 73] 
        0th iteration: time elapsed = 0.13573622703552246 s
Iteration 1: 10.0 %
         medoids = [221 256  25 163]
         labels.freq = [59, 47, 121, 73] 
        1th iteration: time elapsed = 0.08876228332519531 s
# Converged at 1th iteration
...Converged
...Clustering Done (Time elapsed: 0.227003812789917 s)
--------Saving medoids and labels...-------
...Saving Done
```

The input points are distributed like:




