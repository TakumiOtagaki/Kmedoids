# python Kmedoids.py --num_core 64 --input_distmat dist.csv --dist_type (triu|tril|sym) \
#                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000

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
    parser.add_argument('--num_thread', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--input_distmat', type=str, default='test.triu.distmat.csv',
                        help='Input distance matrix')
    parser.add_argument('--dist_type', type=str, default='triu',
                        help='Input distance matrix type (triu|tril|sym)')
    parser.add_argument('--output_medoids', type=str, default='out_medoids.csv',
                        help='Output medoids')
    parser.add_argument('--output_label', type=str, default='out_labels.csv',
                        help='Output labels')
    parser.add_argument('--num_clusters', type=int, default=2,
                        help='Number of clusters')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode', default=False)
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum number of iterations')
    return parser.parse_args()

def read_distmat(distmat_file, dist_type):
    # read distance matrix from distance_matrix.csv
        # dist_type: triu (upper triangle), tril (lower triangle), sym (symmetry)
    # return: distance matrix

    if dist_type not in ['triu', 'tril', 'sym']:
        raise ValueError('dist_type must be triu, tril, or sym')
    if not os.path.exists(distmat_file):
        raise ValueError('distance matrix file not found')

    # avoid using pandas.read_csv to save memory, use 
    # np.loadtxt instead
    distmat = np.loadtxt(distmat_file, delimiter=',')

    # convert to symmetry distance matrix
    if dist_type == 'triu':
        distmat = np.triu(distmat) + np.triu(distmat).T
    elif dist_type == 'tril':
        distmat = np.tril(distmat) + np.tril(distmat).T
    elif dist_type == 'sym':
        pass
    
    return distmat

def update_medoids(distmat, labels, medoids, i):
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


def kmedoids(distmat, num_clusters, num_thread, verbose, max_iter=300):
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

    # start kmedoids

    for iter in range(max_iter):
        if verbose and iter % 100 == 0:
            print(f"Iteration {iter}: {iter * 1.0 / args.max_iter * 100} %")
        # update medoids
        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results = [pool.apply_async(update_medoids, args=(distmat, labels, medoids, i)) for i in range(num_clusters)]
        medoids_new = [p.get() for p in results]
        # for i in range(num_clusters):
        #     cluster = np.where(labels == i)[0]
        #     distmat_sub = distmat[cluster][:, cluster]
        #     medoids[i] = cluster[np.argmin(np.sum(distmat_sub, axis=1))]

        labels_old = labels.copy()

        # update labels
        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results = [pool.apply_async(np.argmin, args=(distmat[i, medoids_new],)) for i in range(distmat.shape[0])]
        labels = [p.get() for p in results]

        # check convergence
        if np.array_equal(labels, labels_old):
            if verbose: print('Converged')
            break

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
    return True

def main():
    # parse arguments
    args = parse_args()

    # read distance matrix
    if args.verbose: print('Reading distance matrix...')
    distmat = read_distmat(args.input_distmat, args.dist_type)
    if args.verbose: print('Done'); print(f"Distance matrix shape: {distmat.shape}")

    # kmedoids clustering
    medoids, labels = kmedoids(distmat, args.num_clusters, args.num_thread, args.verbose, args.max_iter)

    # save medoids and labels
    np.savetxt(args.output_medoids, medoids, fmt='%d', delimiter=',')
    np.savetxt(args.output_label, labels, fmt='%d', delimiter=',')

if __name__ == '__main__':
    main()