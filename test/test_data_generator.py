# test data となる点の座標を作成する。
# そして続いて distance matrix も計算する

import numpy as np
import matplotlib.pyplot as plt

def points_generator(n, dim, seed, cluster_num):
    """
    n: number of points
    dim: dimension of points
    seed: seed
    cluster_num: number of clusters (center of the distribution)
    """
    np.random.seed(seed)
    points = np.zeros((n, dim))
    points_num_dist = np.random.multinomial(n, [1/cluster_num]*cluster_num)
    center_points_dist = np.random.normal(loc=0, scale=40, size=(cluster_num, dim))
    var_dist = np.random.uniform(low=4, high=10, size=cluster_num)
    for i in range(cluster_num):
        points[sum(points_num_dist[:i]):sum(points_num_dist[:i+1])] = np.random.normal(loc=center_points_dist[i], scale=var_dist[i], size=(points_num_dist[i], dim))
    return points

def distance_matrix(points):
    """
    points: 2d array, shape=(n, dim)
    """
    n = points.shape[0]
    distmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distmat[i, j] = np.linalg.norm(points[i]-points[j])
            distmat[j, i] = distmat[i, j]
    return distmat

def main():
    n, dim, seed = 300, 2, 0
    points = points_generator(n, dim, seed, 5)
    # save points
    np.savetxt(f"test_points_n{n}r{seed}.csv", points, delimiter=",", fmt="%.3f")
    if dim == 2:
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"test points n={n}, seed={seed}")
        plt.savefig(f"./test_points_n{n}r{seed}.png")
    distmat = distance_matrix(points)
    np.savetxt(f"test_distmat_n{n}.csv", distmat, delimiter=",", fmt="%.3f")


if __name__ == "__main__":
    main()