# Paper Information

## Title: I. Azizi, K. Echihabi, T. Palpanas. Elpis: Graph-Based Similarity Search for Scalable Data Science.

## Abstract:

The recent popularity of learned embeddings has fueled the growth of massive collections of high-dimensional (high-d) vectors that model complex data, including images, text, tables, graphs and data series in a variety of scientific and business domains. Finding similar vectors in these collections is at the core of many important and practical data science applications, such as information retrieval, clustering, recommendation, malware detection and data cleaning. The data series community has developed tree-based similarity search techniques that outperform state-of-the-art methods on large collections of both data series and generic high-d vectors, on all scenarios except for no-guarantees (ng) approximate search, where graph-based approaches designed by the high-d vector community achieve the best performance. However, building graph-based indexes is extremely expensive both in time and space. In this paper, we bring these two worlds together, study the corresponding solutions and their performance behavior, and propose Elpis, a new strong baseline that takes advantage of the best features of both to achieve a superior performance in terms of indexing and ng-approximate search in-memory. Elpis builds the index 3x-8x faster than competitors, using 40% less memory. It also achieves a high recall of 0.99, up to 2x faster than the state-of-the-art methods on most scenarios, and answers 1-NN queries up to one order of magnitude faster. We demonstrate the scalability of Elpis on large real datasets from different domains, including computer vision, deep-network embeddings, neuroscience, and seismology, and share key insights useful for further developments and data science applications, supported by a thorough experimental evaluation.

## Paper Link: https://helios2.mi.parisdescartes.fr/~themisp/publications/elpis.pdf

# Datasets

You can download all real datasets following the links in ./data/real/README.md. Please note that all datasets should only contain raw data, the dataset size and dimensions should be given as an input in the program args.

# Building Instruction

## Prerequisites

+ GCC 4.9+ with OpenMP
+ CMake 2.8+
+ Boost 1.55+

## Compile On Ubuntu

1. Install Dependencies:

```shell
$ sudo apt-get install g++ cmake libboost-dev 
```

2. Compile ELPIS:

```shell
$ cd code; chmod 777 *; ./release.sh
```

## Run

### Buiding

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

+ data: Path to dataset to be indexed.
+ size: Size of dataset to be indexed.
+ index_path: The path where ELPIS is supposed to create the index.
+ dimensions: The data dimension.
+ leaf_size: The max leaf size.
+ K: Maximum Outdegree for hierarchical layers in leafgraphs(M，outdegree of base graph is x2).
+ bw: Beamwidth used during graphs building.（efConstruction）
+ buffer-size: Maximum memory size(GB) to by Hercules.

<!-- sift10K -->

$ ./build/ELPIS
--dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_base.bin
--dataset-size 10000
--index-path /home/xln/elpis/index/siftsmall_build/
--timeseries-size 128
--leaf-size 1000
--kb 14
--Lb 32
--buffer-size 512
--mode 0

```shell
$ ./build/ELPIS \
--dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base.bin \
--dataset-size 1000000 \
--index-path /home/xln/elpis/index/sift_2k_delete/ \
--timeseries-size 128 \
--leaf-size 2000 \
--kb 20 \
--Lb 36 \
--buffer-size 2048 \
--mode 0 
```



### Search

```shell
$ ./Release/ELPIS \
--queries queries \
--queries-size size \
--index-path index_path \
--k k \
--L bw \
--nprobes maxvl \
--mode 1 \
```

+ queries: Path to the queries set.
+ size: Size of queries set.
+ index_path: The path where ELPIS index can be found.
+ k: Top-k nearest neighbors answers for each query.
+ bw: Beamwidth used during graphs search.
+ maxvl: Maximum number of leaves to search for each query.

<!-- sift1M -->
### generate leaf

```shell
./build/ELPIS \
--dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_base.bin \
--dataset-size 10000 \
--query_dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_query.bin \
--query_dataset_size 100 \
--groundtruth_dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_groundtruth.bin \
--groundtruth_dataset_size 100 \
--groundtruth_top_k 10 \
--index-path /home/xln/elpis/index/siftsmall_knn10/ \
--timeseries-size 128 \
--leaf-size 1000 \
--mode 2
```

```shell
./build/ELPIS \
--dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_base.bin \
--dataset-size 1000000 \
--query_dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_query.bin \
--query_dataset_size 1000 \
--groundtruth_dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_groundtruth.bin \
--groundtruth_dataset_size 1000 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/gist_query_20K_multK/ \
--timeseries-size 960 \
--leaf-size 20000 \
--kb 16 \
--Lb 400 \
--mode 2
```

```shell
./build/ELPIS \
--dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_base.bin \
--dataset-size 10000 \
--query_dataset home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_query.bin \
--query_dataset_size 100 \
--groundtruth_dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_learn_top100_groundtruth.bin \
--groundtruth_dataset_size 100 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/siftsmall_query_5H_multK/ \
--timeseries-size 128 \
--leaf-size 500\
--kb 16 \
--Lb 400 \
--mode 2
```


``` shell
./build/ELPIS \
--dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base.bin \
--dataset-size 1000000 \
--query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query.bin \
--query_dataset_size 10000 \
--groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
--groundtruth_dataset_size 10000 \
--groundtruth_top_k 100 \
--index-path /home/xln/elpis/index/sift_query_5k_delete/ \
--timeseries-size 128 \
--leaf-size 5000 \
--kb 16 \
--Lb 400 \
--mode 2
```

<!-- learn-top100 -->

```shell

./build/ELPIS \
  --dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_base.bin \
  --dataset-size 10000 \
  --query_dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_merged_query.bin \
  --query_dataset_size 25100 \
  --groundtruth_dataset /home/xln/elpis/data/real/siftsmall/sift10K/bin/siftsmall_merged_groundtruth.bin \
  --groundtruth_dataset_size 25100 \
  --groundtruth_top_k 100 \
  --index-path /home/xln/elpis/index/siftsmall_test/ \
  --timeseries-size 128 \
  --leaf-size 1000 \
  --mode 2


./build/ELPIS \
  --dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_base.bin \
  --dataset-size 1000000 \
  --query_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_query.bin \
  --query_dataset_size 10000 \
  --groundtruth_dataset /home/xln/elpis/data/real/sift1M/sift/bin/sift_groundtruth.bin \
  --groundtruth_dataset_size 10000 \
  --groundtruth_top_k 100 \
  --index-path /home/xln/elpis/index/sift1M_query/ \
  --timeseries-size 128 \
  --leaf-size 10000 \
  --mode 2


./build/ELPIS \
  --dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_base.bin \
  --dataset-size 1000000 \
  --query_dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_merged_query.bin \
  --query_dataset_size 501000 \
  --groundtruth_dataset /home/xln/elpis/data/real/gist1M/gist/bin/gist_merged_groundtruth.bin \
  --groundtruth_dataset_size 501000 \
  --groundtruth_top_k 100 \
  --index-path /home/xln/elpis/index/gist1M/ \
  --timeseries-size 128 \
  --leaf-size 10000 \
  --mode 2

```
