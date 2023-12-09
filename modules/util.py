def available_cpu():
    # check available cpu
    # return: number of available cpu
    return mp.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(description='Kmedoids clustering')
    # help
    parser.add_argument('-c', '--av_cpu', default=False, action='store_true',
                        help='Checking the available CPU. If True, the program will exit after checking available CPU')
    parser.add_argument('-p','--num_thread', type=int, default=1,
                        help='Number of threads. if num_thread > num_points, set num_thread = num_points for avoiding useless cpu usage')
    parser.add_argument('-s','--input_sep', type=str, default=',',
                        help='Input distance matrix separator')
    parser.add_argument('-I','--input_distmat', type=str, default='test.triu.distmat.csv',
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



