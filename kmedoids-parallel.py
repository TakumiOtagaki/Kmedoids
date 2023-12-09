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
from math import ceil
import multiprocessing as mp
from modules.util import parse_args, read_distmat, input_validation


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

def candidate_medoids_parallel(distmat_sub_split, j):
    # 1 <= thread_id <= num_thread

    # return the index of the data point which has the minimum sum of distance to other data points in the cluster in each thread
    # return integer

    # return cluster[np.argmin(distmat_sub_sum)]

    return np.argmin(np.sum(distmat_sub_split[j], axis=1))


def better_medoids_initialization(distmat, num_clusters, verbose, random_seed):
    # initialize medoids randomly, but better than random
        # distmat: distance matrix (symmetry), ndarray
        # num_clusters: number of clusters
    # return: medoids
    # key: choose the first medoid randomly, then choose the rest medoids based on the distance to the medoid before it

    # initialize medoids randomly
    medoids = np.random.randint(distmat.shape[0], size=1)
    for i in range(num_clusters - 1):
        distmat_sub = distmat[:, medoids]
        distmat_sub = np.min(distmat_sub, axis=1)
        medoids = np.append(medoids, np.argmax(distmat_sub))
    return medoids

def medoids_initialization(distmat, num_clusters, verbose, random_seed):
    # initialize medoids randomly
        # distmat: distance matrix (symmetry), ndarray
        # num_clusters: number of clusters
    # return: medoids

    # initialize medoids randomly
    medoids = random.sample(range(distmat.shape[0]), num_clusters)
    return medoids

def kmedoids_iter(distmat, num_clusters, num_thread, verbose, medoids, labels):
    # 3 cases:
        # num_thread = num_clusters
        # num_thread < num_clusters
        # num_thread > num_clusters
    # return: medoids, labels

    labels_old = labels.copy()

    # ------------------------------------ update medoids ------------------------------------
    if num_thread == num_clusters:
        print("num_thread == num_clusters")
        # update medoids
        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results = [pool.apply_async(update_medoids, args=(distmat, labels,  i)) for i in range(num_clusters)]
        medoids = np.array([p.get() for p in results])
        if verbose: print("\tmedoids = {}".format(medoids))
        pool.close()
        # print("\tmedoids_new calculated")
    
    elif num_thread < num_clusters:
        # for each cluster i, sort the cluster by np.where(labels == i)[0].shape[0],
        # then update medoids and labels in order and use multiprocessing to speed up
        # rest of the clusters are updated by the last threads (num_thread - num_clusters)

        # update medoids
        # sort the cluster index by np.where(labels == i)[0].shape[0]: large -> small
        cluster_ind_sorted = np.argsort([np.where(labels == i)[0].shape[0] for i in range(num_clusters)])[::-1]

        # use multiprocessing to speed up
        pool = mp.Pool(processes=num_thread)
        results_head = [pool.apply_async(update_medoids, args=(distmat, labels,  cluster_ind_sorted[i])) for i in range(num_thread)]
        results_tail = [pool.apply_async(update_medoids, args=(distmat, labels,  cluster_ind_sorted[i])) for i in range(num_thread, num_clusters)]
        medoids_head = np.array([p.get() for p in results_head])
        medoids_tail = np.array([p.get() for p in results_tail])
        pool.close()
        medoids = np.concatenate((medoids_head, medoids_tail))

    elif num_thread > num_clusters:
        # for each cluster i, sort the cluster by np.where(labels == i)[0].shape[0],
        # then, distribute the threads (num_thread) to the clusters in nice way.
            # e.g. num_thread = 8, num_clusters = 5, then 3 threads are distributed to the clusters with large size.
                # cluster 0: 3 threads
                # cluster 1: 2 threads
                # cluster 2: 1 thread
                # cluster 3: 1 thread
                # cluster 4: 1 thread
        # in this block, we need to update medoids with multi core.

        # update medoids
        # sort the cluster index by np.where(labels == i)[0].shape[0]: large -> small
        cluster_ind_sorted = np.argsort([np.where(labels == i)[0].shape[0] for i in range(num_clusters)])[::-1]

        # use multiprocessing to speed up
        
        # thread distribution: 
            # is proportion to np.where(labels == i)[0].shape[0] (cluster size)
            # each cluster has at least 1 thread

        thread_distribution = np.zeros(num_clusters, dtype=np.int32)

        # proportion to np.where(labels == i)[0].shape[0] (cluster size)
        cluster_size = np.array([np.where(labels == i)[0].shape[0] for i in range(num_clusters)])
        cluster_size = cluster_size / np.sum(cluster_size) # 0 <= cluster_size <= 1
        thread_distribution = np.ones(num_clusters, dtype=np.int32) + (cluster_size * (num_thread - num_clusters)).astype(np.int32)
        # add the rest of the threads to the clusters with large size
        thread_distribution[cluster_ind_sorted[:num_thread - np.sum(thread_distribution)]] += 1


        print("thread_distribution = {}".format(thread_distribution))
        print("np.sum(thread_distribution) = {}".format(np.sum(thread_distribution)))
        # print(thread_distribution)
        pool = mp.Pool(processes=num_thread)
        for k in range(num_clusters):
            # get the argmin candidate.
            # use multiprocessing to speed up
            cluster = np.where(labels == k)[0]
            distmat_sub = distmat[cluster][:, cluster]
            distmat_sub_split = np.array_split(distmat_sub, thread_distribution[k])
            results_k = [pool.apply_async(candidate_medoids_parallel, args=(distmat_sub_split, j)) for j in range(thread_distribution[k])]
            medoids[k] = cluster[np.array([p.get() for p in results_k]).argmin()]
        pool.close()
        if verbose: print("\tmedoids_new calculated")

        # update labels


    # ------------------------------------ update labels ------------------------------------
    # use multiprocessing to speed up
    pool = mp.Pool(processes=num_thread)
    # distribute the threads to distmat.shape[0].
    # distmat.shape[0] is larger than num_thread, so we need to split distmat.shape[0] into num_thread.
    # calc np.argmin, args=(distmat[i, medoids] for i in range(distmat.shape[0])).
    # for i in range(distmat.shape[0]), we need to calc np.argmin, args=(distmat[i, medoids]). split this into num_thread.


    for i in range(ceil(distmat.shape[0] / num_thread)):
        results = [pool.apply_async(np.argmin, args=(distmat[i * num_thread + j, medoids],)) for j in range(num_thread)]
        labels[i * num_thread : (i + 1) * num_thread] = np.array([p.get() for p in results])

    if verbose: print("\tlabel_new calculated")
    return medoids, labels_old, labels


def kmedoids(distmat, num_clusters, num_thread, verbose, max_iter, random_seed):
    # kmedoids clustering
        # distmat: distance matrix (symmetry), ndarray
        # num_clusters: number of clusters
        # num_thread: number of threads
    # return: medoids, labels

    # initialize medoids randomly
    # medoids = random.sample(range(distmat.shape[0]), num_clusters)
    medoids = better_medoids_initialization(distmat, num_clusters, verbose, random_seed)
    labels = np.zeros(distmat.shape[0], dtype=np.int32)
    for i in range(distmat.shape[0]):
        labels[i] = np.argmin(distmat[i, medoids])

    if verbose: print('Initialization done'); iter_span = (max_iter // 5) if max_iter > 5 else 1

    # start kmedoids
    for iter in range(max_iter):
        if verbose:
            start = time.time()
            print(f"Iteration {iter}: {iter * 1.0 / max_iter * 100} %")

        medoids, labels_old, labels = kmedoids_iter(distmat, num_clusters, num_thread, verbose, medoids, labels)

        # check convergence
        if np.array_equal(labels, labels_old):
            if verbose: print(f'Converged at {iter}th iteration')
            break
        if verbose: print(f"Time elapsed: {time.time() - start} s for {iter}th iteration")
    if verbose: print('Not converged')
    return medoids, labels


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