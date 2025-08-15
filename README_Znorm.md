The time series in both datasets were normalized with Z-normalization.


1. Index build

```shell
$ ./Release/ELPIS \
--dataset data \
--dataset-size size \
--index-path index_path \
--timeseries-size dimensions \
--leaf-size leaf_size \
--kb K  \
--Lb bw \
--mode 0 \
--buffer-size MaxGB

```

2. Index Search

```shell
$ ./Release/ELPIS \
--queries query dataset \
--queries-size size of query dataset \
--groundtruth_dataset Groundtruth of query && calculate Recall\
--groundtruth_dataset_size Verification function: groundtruth_dataset_size==queries-size\
--groundtruth_top_k \
--learn_dataset To train weight \
--learn_dataset_size Size of learn_dataset \
--learn_groundtruth_dataset  Groundtruth of query of learn dataset\
--index-path Index_path \
--nprobes  Number of leaf nodes to be searched\
--ep Weight Iteration Count\
--thres_probability  \
--μ Para to calculate probability \
--T Para to calculate probability \
--zero_edge_pass_ratio  The probability to skip zero-weight edge\
--k k  top-k results\
--L bw  efSearch\
--mode 1 \

```

``` shell
./build/ELPIS \
--dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm.bin \
--dataset-size 1000000 \
--query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query.bin \
--query_dataset_size 10000 \
--groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
--groundtruth_dataset_size 10000 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/sift_10k_znorm/ \
--timeseries-size 128 \
--leaf-size 10000 \
--kb 16 \
--Lb 400 \
--mode 2
```


组合1
''' shell
<!-- sift1M -->
$ ./build/ELPIS \
  --dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm.bin \
  --dataset-size 1000000 \
  --index-path /home/xln/elpis/index/sift_10k_znorm/ \
  --timeseries-size 128 \
  --leaf-size 10000 \
  --kb 20 \
  --Lb 36 \
  --buffer-size 512 \
  --mode 0


  
<!-- sift1M -->
./build/ELPIS \
  --query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query_znorm.bin \
  --query_dataset_size 10000 \
  --groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
  --groundtruth_dataset_size 10000 \
  --groundtruth_top_k 100 \
  --learn_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_learn_znorm.bin \
  --learn_dataset_size 100000 \
  --learn_groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_learn_top100_groundtruth.bin \
  --dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm.bin \
  --index-path /home/xln/elpis/index/sift_10k_znorm/ \
  --nprobes 10 \
  --ep 0 \
  --thres_probability 0.1 \
  --μ -2.197 \
  --T 1.0 \
  --zero_edge_pass_ratio 0 \
  --k 10 \
  --L 24 \
  --mode 1


./build/ELPIS \
--dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm.bin \
--dataset-size 1000000 \
--query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query.bin \
--query_dataset_size 10000 \
--groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
--groundtruth_dataset_size 10000 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/sift_10k_znorm/ \
--timeseries-size 128 \
--leaf-size 10000 \
--kb 16 \
--Lb 400 \
--mode 2
'''


组合2
''' shell
<!-- sift1M -->
$ ./build/ELPIS \
  --dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm_row.bin \
  --dataset-size 1000000 \
  --index-path /home/xln/elpis/index/sift_5k_znorm_row/ \
  --timeseries-size 128 \
  --leaf-size 5000 \
  --kb 20 \
  --Lb 36 \
  --buffer-size 512 \
  --mode 0


  
<!-- sift1M -->
./build/ELPIS \
  --query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query_znorm_row.bin \
  --query_dataset_size 10000 \
  --groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
  --groundtruth_dataset_size 10000 \
  --groundtruth_top_k 100 \
  --learn_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_learn_znorm_row.bin \
  --learn_dataset_size 100000 \
  --learn_groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_learn_top100_groundtruth.bin \
  --dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm_row.bin \
  --index-path /home/xln/elpis/index/sift_10k_znorm_row/ \
  --nprobes 10 \
  --ep 0 \
  --thres_probability 0.1 \
  --μ -2.197 \
  --T 1.0 \
  --zero_edge_pass_ratio 0 \
  --k 10 \
  --L 24 \
  --mode 1


./build/ELPIS \
--dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base_znorm_row.bin \
--dataset-size 1000000 \
--query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query_row.bin \
--query_dataset_size 10000 \
--groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
--groundtruth_dataset_size 10000 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/sift_10k_znorm_row/ \
--timeseries-size 128 \
--leaf-size 10000 \
--kb 16 \
--Lb 400 \
--mode 2
'''