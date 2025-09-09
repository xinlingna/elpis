#ifndef HERCULES_H
#define HERCULES_H

#include "Index.h"
#include "Node.h"
#include "file_utils.h"
#include "QueryEngine.h"
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <set>

class Hercules
{
public:
    // Member variables
    Index *index;
    Node **leaves;
    int num_leaf_node;
    int leaf_size;

    std::unordered_map<file_position_type, Node *> ts_leaf_map; // Mapping from time series ID to leaf node
    std::unordered_map<int, int> leafId2Idx; // Mapping from leaf ID to index
    std::vector<std::vector<int>> leaf_topk_indices; // Mapping from leaf ID to top-k indices

    char *dataset;
    int dataset_size;
    VectorWithIndex *ts_list;

    char *groundtruth_dataset;
    unsigned int groundtruth_dataset_size;
    int **groundtruth_list; 
    unsigned int groundtruth_top_k;
    int **learn_groundtruth_list;

    int timeseries_size;
    char *index_path;
    file_position_type **knn_distributions;
    file_position_type **knn_groundtruth;


    leaf_centroid *centroids_centroids;
    leaf_centroid *centroids_center;

    const char* statistics_path = "/home/xln/elpis/data/statistics/";

    // Constructor
/*     Hercules(char *dataset, int dataset_size,
             char *query_dataset, int query_dataset_size,
             char *groundtruth_dataset, int groundtruth_dataset_size, int groundtruth_top_k,
             char *learn_dataset, int learn_dataset_size, char* learn_groundtruth_dataset,
             char *index_path, int timeseries_size, int leaf_size,
            int construction, int m, int mode); */
    Hercules(char *dataset, unsigned int dataset_size,
			 char *query_dataset, unsigned int query_dataset_size,
			 char *groundtruth_dataset, unsigned int groundtruth_dataset_size,unsigned int groundtruth_top_k, 
			 char * learn_dataset, unsigned int learn_dataset_size, char * learn_groundtruth_dataset,
			 char *index_path, unsigned int timeseries_size, unsigned int leaf_size,
             unsigned int nprobes, bool parallel, unsigned int nworker, bool flatt,
			 int efConstruction, unsigned int m, int efSearch, unsigned int k, unsigned int ep,
			 char *model_file, float zero_edge_pass_ratio, 
             bool search_withWeight, float μ, float T, float thres_probability,int mode);

    // Function declarations
    void buildIndexTree();
    void generateLeafId2IdxMap(const char* output_path = nullptr);
    void leafIndexToIdMap();
    void writeLeafId2IdxFile(const char* output_path_);
    ts_type* generate_segment(Node* node, size_t* segments_dimension);
    void generate_leafnode_file();
    // void generate_leaf_centroids(const char* output_path = nullptr, const char* generate_way = "centroid");
    void generate_leaf_centroids(const char* output_path = nullptr);
    void write_centroid_file(const char* method, leaf_centroid* centroids);

    void generate_ts_leaf_map_file(const char* output_path = nullptr);
    void fillTsLeafMap();
    void writeTsLeafMapFile(const char* output_path_);
    void topk2LeafId(const char* output_path = nullptr);
    void leafContainsTopK(int selected_k);
    void generate_label(unsigned int selected_k=100, const char* output_path = nullptr);
    void calcKNNinLeaves(unsigned int selected_k);
    void writeKNNDistributionsToFile(unsigned int selected_k,const char* output_path_=nullptr);

    // edge weight

    // Index * index;
    int mode; 
    int efSearch;
    unsigned int nprobes;
    bool parallel;
    unsigned int nworker;
    bool flatt;
    unsigned int k; // top-k
    unsigned int ep; // epoch
    float thres_probability; 
    float μ;
    float T;
    bool search_withWeight; // 是否使用权重搜索


    char * model_file;
    float zero_edge_pass_ratio;

    char * query_dataset;  unsigned int query_dataset_size;
    char * learn_dataset;  unsigned int learn_dataset_size;
    char * learn_groundtruth_dataset;

    QueryEngine *queryengine;

    std::vector<std::set<Node*>> learn_candidate_leaf_node;   // candidate leaf nodes per learn vector
    std::vector<std::set<Node*>> query_candidate_leaf_node;   // candidate leaf nodes per query vector
    std::unordered_map<int, Node*> leafNode2GraphMap; // map leafnode->id to graph
    void TrainWeight();
    void getLearnCandidateLeafNode(unsigned int selected_k);
    void getQueryCandidateLeafNode(unsigned int selected_k);

    void trainCandidateLeafNode();
    void searchCandidateLeafNode();
    void leafNodeID2Node();

    // 写出叶子包含的向量数目，顺序与质心一致（leafId2Idx映射后的顺序）
    void write_leaf_sizes_file(const char* output_path_ = nullptr);


    void generateAllFiles();
    // Generate additional per-leaf cluster info files (sizes, variances, densities,
    // intra-distances and representative vectors). Files are written so their
    // line order corresponds exactly to the centroid order produced by
    // `generate_leaf_centroids()` (i.e. mapped by `leafId2Idx`).
    void generate_cluster_info_files(const char* output_path = nullptr, int num_representatives = 3);
    void generate_cluster_info_files_corrected(const char* output_path = nullptr);
    // Destructor
    ~Hercules();

private:
    // Private methods and members (if any)
};

#endif // HERCULES_H
