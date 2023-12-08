# Takumi Otagaki
# 2023/12/08

# python Kmedoids.py --num_threads 4 --input_distmat dist.csv --dist_type (triu|tril|sym) \
#                    --input_sep "," \
#                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000 \
#                    --verbose --random_seed 0

import argparse
import numpy as np
import random
import time
import sys
import os
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description='Kmedoids clustering')
    # help
    parser.add_argument('-c', '--av_cpu', default=False, action='store_true',
                        help='Check available CPU. If True, the program will exit after checking available CPU')
    parser.add_argument('-p','--num_thread', type=int, default=1,
                        help='Number of threads. if num_thread > num_points, set num_thread = num_points for avoiding useless cpu usage')
    parser.add_argument('-s','--input_sep', type=str, default=',',
                        help='Input distance matrix separator')
    parser.add_argument( '-I','--input_distmat', type=str, default='test.triu.distmat.csv',
                        help='Input distance matrix')
    parser.add_argument('-T', '--dist_type',  type=str, default='triu',
                        help='Input distance matrix type (triu|tril|sym)')
    parser.add_argument('-M','--output_medoids',  type=str, default='out_medoids.csv',
                        help='Output medoids')
    parser.add_argument('-L', '--output_label',  type=str, default='out_labels.csv',
                        help='Output labels')
    parser.add_argument( '-k','--num_clusters', type=int, default=2,
                        help='Number of clusters')
    parser.add_argument('-v', '--verbose',  action='store_true',
                        help='Verbose mode', default=False)
    parser.add_argument('-N','--max_iter',  type=int, default=300,
                        help='Maximum number of iterations')
    parser.add_argument('-r','--random_seed', type=int, default=0,
                        help='Random seed: Should be integer')
    return parser.parse_args()

def read_distmat(distmat_file, dist_type, sep):
    # read distance matrix from distance_matrix.csv
        # dist_type: triu (upper triangle), tril (lower triangle), sym (symmetry)
    # return: distance matrix

    if dist_type not in ['triu', 'tril', 'sym']:
        raise ValueError('dist_type must be triu, tril, or sym')
    if not os.path.exists(distmat_file):
        raise ValueError('distance matrix file not found')

    # avoid using pandas.read_csv to save memory, use 
    # np.loadtxt instead
    distmat = np.loadtxt(distmat_file, delimiter=sep)

    # convert to symmetry distance matrix
    if dist_type == 'triu':
        distmat = np.triu(distmat) + np.triu(distmat).T
    elif dist_type == 'tril':
        distmat = np.tril(distmat) + np.tril(distmat).T
    elif dist_type == 'sym':
        pass
    
    return distmat

def update_medoids(distmat, labels,  i):
    # update medoids
        # distmat: distance matrix (symmetry), ndarray
        # labels: labels of each data point, ndarray
        # medoids: medoids, list
        # i: cluster index
    # return: medoids[i]
    
    cluster = np.where(labels == i)[0]
    distmat_sub = distmat[cluster][:, cluster]
    # medoids[i] = cluster[np.argmin(np.sum(distmat_sub, axis=1))]
    # return medoids[i]
    return cluster[np.argmin(np.sum(distmat_sub, axis=1))]


def kmedoids(distmat, num_clusters, num_thread, verbose, max_iter, random_seed):
    # kmedoids clustering
        # distmat: distance matrix (symmetry), ndarray
        # num_clusters: number of clusters
        # num_thread: number of threads
    # return: medoids, labels

    # initialize medoids randomly
    medoids = random.sample(range(distmat.shape[0]), num_clusters)
    labels = np.zeros(distmat.shape[0], dtype=np.int32)
    for i in range(distmat.shape[0]):
        labels[i] = np.argmin(distmat[i, medoids])

    if verbose: print('Initialization done'); iter_span = (max_iter // 5) if max_iter > 5 else 1

    # start kmedoids
    
    for iter in range(max_iter):
        if verbose:
            start = time.time()
            print(f"Iteration {iter}: {iter * 1.0 / max_iter * 100} %")
        # update medoids
        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results = [pool.apply_async(update_medoids, args=(distmat, labels,  i)) for i in range(num_clusters)]
        medoids = np.array([p.get() for p in results])
        pool.close()
        print("\tmedoids_new calculated")

        labels_old = labels.copy()

        # update labels
        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results = [pool.apply_async(np.argmin, args=(distmat[i, medoids],)) for i in range(distmat.shape[0])]
        labels = np.array([p.get() for p in results])
        pool.close()
        print("\tlabel_new calculated")

        # check convergence
        if np.array_equal(labels, labels_old):
            if verbose: print('Converged')
            break
        if verbose: print(f"Time elapsed: {time.time() - start} s for {iter}th iteration")
    if verbose: print('Not converged')
    return medoids, labels

def input_validation(args):
    # input validation
    if args.num_thread <= 0:
        raise ValueError('num_thread must be greater than 0')
    if args.dist_type not in ['triu', 'tril', 'sym']:
        raise ValueError('dist_type must be triu, tril, or sym')
    if not os.path.exists(args.input_distmat):
        raise ValueError('distance matrix file not found')
    if args.num_clusters <= 0:
        raise ValueError('num_clusters must be greater than 0')
    if args.max_iter <= 0:
        raise ValueError('max_iter must be greater than 0')
    if args.num_points <= 0:
        raise ValueError('num_points must be greater than 0')
    if args.num_points > args.num_thread:
        raise ValueError('num_points must be less than or equal to num_thread')
    if type(args.random_seed) != int:
        raise ValueError('random_seed must be integer')
    return True

def available_cpu():
    # check available cpu
    # return: number of available cpu
    return mp.cpu_count()

def main():
    # parse arguments
    args = parse_args()

    if args.av_cpu: print(f"Available CPU: {available_cpu()}\nexit.") ; return
    
    # read distance matrix
    if args.verbose: print('Reading distance matrix...'); start = time.time()
    distmat = read_distmat(args.input_distmat, args.dist_type, args.input_sep)
    if args.verbose: print('Done'); print(f"Distance matrix shape: {distmat.shape}"); print(f"Time elapsed: {time.time() - start} s")

    # if args.num_points > args.num_thread: args.num_thread = args.num_points
    if distmat.shape[0] < args.num_thread: 
        args.num_thread = distmat.shape[0]
        print("Warning: num_points > num_thread, set num_thread = num_points")

    # kmedoids clustering
    if args.verbose: print('Clustering...'); start = time.time()
    medoids, labels = kmedoids(distmat, args.num_clusters, args.num_thread, args.verbose, args.max_iter, args.random_seed)
    if args.verbose: print('Done'); print(f"Time elapsed: {time.time() - start} s")

    # save medoids and labels
    if args.verbose: print('Saving medoids and labels...'); start = time.time()
    np.savetxt(args.output_medoids, medoids, fmt='%d', delimiter=',')
    np.savetxt(args.output_label, labels, fmt='%d', delimiter=',')
    if args.verbose: print('Done'); print(f"Time elapsed: {time.time() - start} s")
    

if __name__ == '__main__':
    main()