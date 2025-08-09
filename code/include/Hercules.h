#ifndef HERCULES_H
#define HERCULES_H

#include "Index.h"
#include "Node.h"
#include "file_utils.h"
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>

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
    int groundtruth_dataset_size;
    int **groundtruth_list; // Assuming groundtruth is a list of indices or similar
    int groundtruth_top_k;

    int timeseries_size;
    char *index_path;
    file_position_type **knn_distributions;
    file_position_type **knn_groundtruth;

    leaf_centroid *centroids_centroids;
    leaf_centroid *centroids_center;

    const char* statistics_path = "/home/xln/elpis/data/statistics/";

    // Constructor
    Hercules(char *dataset, int dataset_size, char *index_path, int timeseries_size, int leaf_size,
             char *query_dataset, int query_dataset_size,
             char *groundtruth_dataset, int groundtruth_dataset_size, int groundtruth_top_k,
            int construction, int m);

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
    void generate_label(const char* output_path = nullptr);
    void calcKNNinLeaves();
    void writeKNNDistributionsToFile(const char* output_path_);


    void generateAllFiles();
    
    // Destructor
    ~Hercules();

private:
    // Private methods and members (if any)
};

#endif // HERCULES_H
