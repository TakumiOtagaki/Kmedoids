# python Kmedoids.py --num_thread 64 --input_distmat dist.csv --dist_type (triu|tril|sym) \
#                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2

import argparse
import numpy as np
import random
import time
import sys
import os
import multiprocessing as mp
from functools import partial


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

def kmedoids(distmat, num_clusters, num_thread):
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
    while True:
        # update medoids
        for i in range(num_clusters):
            cluster = np.where(labels == i)[0]
            distmat_sub = distmat[cluster][:, cluster]
            medoids[i] = cluster[np.argmin(np.sum(distmat_sub, axis=1))]

        # update labels
        labels_old = labels.copy()
        for i in range(distmat.shape[0]):
            labels[i] = np.argmin(distmat[i, medoids])

        # check convergence
        if np.array_equal(labels, labels_old):
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
    distmat = read_distmat(args.input_distmat, args.dist_type)
    print('distance matrix shape:', distmat.shape)

    # kmedoids clustering
    medoids, labels = kmedoids(distmat, args.num_clusters, args.num_thread)

    # save medoids and labels
    np.savetxt(args.output_medoids, medoids, fmt='%d', delimiter=',')
    np.savetxt(args.output_label, labels, fmt='%d', delimiter=',')

if __name__ == '__main__':
    main()
