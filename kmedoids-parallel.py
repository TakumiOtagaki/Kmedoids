# Takumi Otagaki
# 2023/12/08

# python Kmedoids.py --num_threads 4 --input_distmat dist.csv --dist_type (triu|tril|sym) \
#                    --input_sep "," \
#                    --output_medoids medoids.csv --output_label labels.csv --num_clusters 2 --max_iter 1000 \
#                    --verbose --random_seed 0

import numpy as np
import random
import time
import sys
from math import ceil, floor
import multiprocessing as mp
from modules.util import parse_args, read_distmat, input_validation, available_cpu, printvb
from functools import partial


def update_medoids(distmat, labels,  i):
    # update medoids
    # distmat: distance matrix (symmetry), ndarray
    # labels: labels of each data point, ndarray
    # medoids: medoids, list
    # i: cluster index
    # return: medoids[i]

    cluster = np.where(labels == i)[0]
    distmat_sub = distmat[cluster][:, cluster]
    # return medoids[i] = cluster[np.argmin(np.sum(distmat_sub, axis=1))]
    return cluster[np.argmin(np.sum(distmat_sub, axis=1))]


def better_medoids_initialization(distmat, num_clusters, verbose, random_seed):
    # initialize medoids randomly, but better than random
    # distmat: distance matrix (symmetry), ndarray
    # num_clusters: number of clusters
    # return: medoids
    # key: choose the first medoid randomly, then choose the rest medoids based on the distance to the medoid before it

    # initialize medoids randomly
    np.random.seed(seed=random_seed)
    print = partial(printvb, verbose)
    print("kmedoids++: Better initialization.")
    medoids = np.random.randint(distmat.shape[0], size=1)
    # medoids must be unique.
    # medoids.append(x), where x is furthest from existing medoids.
    for i in range(num_clusters - 1):
        # distmat[medoids]: distance from medoids to all data points
        # distmat[medoids].min(axis=0): distance from medoids to the closest medoid
        # distmat[medoids].min(axis=0).argmax(): index of the data point which is furthest from existing medoids
        medoids = np.append(medoids, distmat[medoids].min(
            axis=0).argmax())
    return medoids


def medoids_initialization(distmat, num_clusters, verbose, random_seed):
    # initialize medoids randomly
    # distmat: distance matrix (symmetry), ndarray
    # num_clusters: number of clusters
    # return: medoids

    # initialize medoids randomly
    if verbose:
        print("kmedoids normal initialization.")
    random.seed(random_seed)
    medoids = random.sample(range(distmat.shape[0]), num_clusters)
    return medoids


def kmedoids_iter(distmat, num_clusters, num_thread, verbose, medoids, labels):
    # return: medoids, labels

    labels_old = labels.copy()
    print = partial(printvb, verbose)

    # ------------------------------------ update medoids ------------------------------------

    # update medoids
    pool = mp.Pool(processes=min(num_thread, num_clusters))
    results = [pool.apply_async(update_medoids, args=(
        distmat, labels,  i)) for i in range(num_clusters)]
    medoids = np.array([p.get() for p in results])
    pool.close()

    # ------------------------------------ update labels ------------------------------------
    # use multiprocessing to speed up
    pool = mp.Pool(processes=num_thread)
    # distribute the threads to distmat.shape[0].
    # distmat.shape[0] is larger than num_thread, so we need to split distmat.shape[0] into num_thread.
    # calc np.argmin, args=(distmat[i, medoids] for i in range(distmat.shape[0])).

    results = [pool.apply_async(np.argmin, args=(
        distmat[i, medoids],)) for i in range(distmat.shape[0])]
    labels = np.array([p.get() for p in results])

    return medoids, labels_old, labels


def kmedoids(distmat, num_clusters, num_thread, verbose, max_iter, random_seed, better_init):
    # kmedoids clustering
    # distmat: distance matrix (symmetry), ndarray
    # num_clusters: number of clusters
    # num_thread: number of threads
    # return: medoids, labels

    # initialize medoids randomly
    converged = False
    print = partial(printvb, verbose)

    if better_init:
        medoids = better_medoids_initialization(
            distmat, num_clusters, verbose, random_seed)
    else:
        medoids = medoids_initialization(
            distmat, num_clusters, verbose, random_seed)

    labels = np.zeros(distmat.shape[0], dtype=np.int32)
    for i in range(distmat.shape[0]):
        labels[i] = np.argmin(distmat[i, medoids])
    print('...Initialization done.')
    print(f"\t medoids = {medoids}")
    print(f"\t labels.freq = {[np.where(labels == i)[0].shape[0] for i in range(num_clusters)]} ")

    # start kmedoids
    print("Main loop starts....")
    for iter in range(max_iter):
        start = time.time()
        print(f"Iteration {iter}: {iter * 1.0 / max_iter * 100} %")

        medoids, labels_old, labels = kmedoids_iter(
            distmat, num_clusters, num_thread, verbose, medoids, labels)


        print(f"\t medoids = {medoids}")
        print(f"\t labels.freq = {[np.where(labels == i)[0].shape[0] for i in range(num_clusters)]} ")
        print(f"\t{iter}th iteration: time elapsed = {time.time() - start} s")
        # check convergence
        if np.array_equal(labels, labels_old):
            print(f'# Converged at {iter}th iteration')
            converged = True
            break


    if converged:
        print('...Converged')
    else:
        print("...Iteration done. (Not converged)")
    return medoids, labels


def main():
    # parse arguments
    args = parse_args()

    if args.av_cpu:
        print(f"Available CPU: {available_cpu()}\nexit.")
        return

    print1 = partial(printvb, args.verbose)
    # read distance matrix
    print1('Reading distance matrix...')
    start = time.time()

    distmat = read_distmat(args.input_distmat, args.dist_type, args.input_sep)
    print1('Reading distmat Done')
    print1(f"\tDistance matrix shape: {distmat.shape}")
    print1(f"\tReading distmat: Time elapsed =  {time.time() - start} s")

    # if args.num_points > args.num_thread: args.num_thread = args.num_points
    if distmat.shape[0] < args.num_thread:
        args.num_thread = distmat.shape[0]
        print1("Warning: num_points > num_thread, set num_thread = num_points")

    # kmedoids clustering
    print1('-------Clustering Starts...-------')
    start = time.time()
    medoids, labels = kmedoids(distmat, args.num_clusters, args.num_thread,
                               args.verbose, args.max_iter, args.random_seed, args.better_init)
    print1(f'...Clustering Done (Time elapsed: {time.time() - start} s)')

    # save medoids and labels
    print1('--------Saving medoids and labels...-------')
    start = time.time()
    np.savetxt(args.output_medoids, medoids, fmt='%d', delimiter=',')
    np.savetxt(args.output_label, labels, fmt='%d', delimiter=',')
    print1('...Saving Done')


if __name__ == '__main__':
    main()
