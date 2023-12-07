# K-medoids

A Python script for K-medoids clustering with multi-process.

# Introduction
There is no useful multiprocessing implementation of K-medoids.

```.sh
python Kmedoids.py --num_thread 64 --input_distmat dist.csv --dist_type (triu|tril|sym) \
 --output_medoids out_medoids.csv --output_label out_labels.csv
```

# usage
For now, this script can be applied when the input distance matrix contains no-integer.