

#ifndef herculesHNSW_QUERYENGINE_H
#define herculesHNSW_QUERYENGINE_H

#include <iostream>
#include <thread>  // 包含 sleep_for 所需的头文件
#include <chrono>  // 包含时间表示的头文件

#include "Index.h"
#include "globals.h"
#include "hnswlib/hnswlib.h"
#include "pqueue.h"
#include <queue>
#include <set>
#include "future"

typedef struct query_result query_result;
typedef unsigned int  IterRefinement_epoch;

typedef struct bsf_snapshot bsf_snapshot;

typedef struct q_index q_index;

typedef struct query_worker_data  worker_backpack__; // 定义结构体类型并为其指定别名

struct CompareByFirst{
    constexpr bool operator()(std::pair<float, unsigned int> const &a,std::pair<float, unsigned int> const &b) const noexcept {
        return a.first < b.first;
    }
};

class QueryEngine {
public :
    QueryEngine();

    QueryEngine(const char *query_filename, unsigned int query_dataset_size, 
                const char *groundtruth_filename,unsigned int groundtruth_top_k,  unsigned int groundtruth_dataset_size,
                const char* learn_dataset, unsigned learn_dataset_size,  const char *learn_groundtruth_dataset,
                const char* dataset,
                Index *index, int ef, unsigned int nprobes, bool parallel,
                unsigned int nworker, bool flatt, unsigned int k, unsigned int ep, const char* model_file=nullptr,
                float zero_edge_pass_ratio = 0.0f);

    Index * index;
    const char * dataset;
    const char * query_filename;
    int query_dataset_size;
    const char *groundtruth_filename;
    unsigned int groundtruth_dataset_size;
    int groundtruth_top_k;

    const char* learn_dataset;
    unsigned int  learn_dataset_size;
    const char* learn_groundtruth_dataset;

    const char * model_file;
    std::vector<std::set<Node*>> candidate_leaf_node;
    float zero_edge_pass_ratio; // ρ: 权重为0的边按该概率放行



    unsigned int nprobes;
    unsigned int ep;
    queue<unsigned int> visited;
    bool parallel;
    unsigned int nworker;
    std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates; // 最大堆
    FILE *query_file;
    FILE *groundtruth_file;
    FILE *learn_groundtruth_file;

    int k;

    querying_stats stats;   // global stats
    // float *results;       
    // std::vector<std::pair<float,unsigned int>> topkID;
    int ** results;
    int ** learn_results;



    worker_backpack__ *qwdata;
    pqueue_t *pq;  
    pqueue_t *candidate_leaves;
    void closeFile();

    file_position_type total_records;



    void queryBinaryFile(unsigned int k, int i, float thres_probability, float μ, float T);

    float calculateRecall(std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> top_candidates, 
                      int* groundtruth_id, unsigned int k, unsigned int groundtruth_top_k);

    double calculateAverageRecall();

    void printKNN(std::vector<std::pair<float,unsigned int>> topkID, int k, double time, queue<unsigned int> &visited, bool para=true);

    void searchNpLeafParallel(ts_type *query_ts, int* groundtruth_id, unsigned int k, unsigned int nprobes, unsigned int query_index, float thres_probability, float μ, float T);
    void searchPredictedNpLeafParallel(ts_type *query_ts, int* groundtruth_id, unsigned int k, unsigned int nprobes, unsigned int query_index);

    void searchGraphLeaf(Node * node,const void *query_data, size_t k,
                     std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                     float &bsf, querying_stats &stats, unsigned short *flags, unsigned short & flag,  bool searchWithWeight=false, float thres_probability=0.3, float μ=0.0, float T=1.0) ;
    void searchflat(Node * node, unsigned int entrypoint, const void *data_point, size_t beamwidth,size_t k,
                std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> & top_candidates,
                float & bsf,querying_stats & stats, unsigned short *threadvisits, unsigned short & round_visit);

    void TrainWeightByLearnDataset(IterRefinement_epoch ep, unsigned int k, std::vector<std::set<Node*>>   candidate_leaf_node);
    void TrainWeightinNpLeafParallel(ts_type *query_ts, int *groundtruth_id, unsigned int k, unsigned int nprobes, unsigned int query_index, IterRefinement_epoch ep, std::set<Node*> candidate_leaf);

    void TrainWeightinGraphLeaf(Node * node,const void *query_data, size_t k,
                     std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                     float &bsf, querying_stats &stats, unsigned short *flags, unsigned short & flag, IterRefinement_epoch ep=0);


    void TrainWeightinflat(Node * node, unsigned int entrypoint, const void *data_point, size_t beamwidth,size_t k,
                std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> & top_candidates1,
                float & bsf,querying_stats & stats, unsigned short *threadvisits, unsigned short & round_visit, IterRefinement_epoch ep);

    void searchflatWithHopPath(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates_internalID,
                float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                std::vector<unsigned int> &hop_path);

    
    void  searchflatWithWeight(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                          float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                          float thres_probability=0.3,float μ=0.0, float T=1.0);

    void searchflatWithWeight_HopPath(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates_internalID,
                          float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                          std::vector<unsigned int> &hop_path, float μ, float T,float thres_probability);
                
    void updateWeightByHopPath( Node *node, const void *data_point, 
                            std::vector<unsigned int> hop_path1, std::vector<unsigned int> hop_path2,
                            std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates1,
                            std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates2);


    void setEF(Node *node, int ef);

    inline void copypq(query_worker_data *pData, priority_queue<pair<float, unsigned int>> queue);
    inline void clearlocalknn(query_worker_data *pData);

    

    ~QueryEngine();



    // 与查询过程中的 状态标记 有关
    unsigned short *flags;
    unsigned short curr_flag;
};


/**
  @param ts_type distance;
  @param struct hercules_node *node;
  @param ts_type max_distance;
  @param size_t pqueue_position;//unsigned long
 */
struct query_result {
    ts_type distance;
    Node *node;
    hnswlib::labeltype ts_num;
    ts_type max_distance;
    size_t pqueue_position;
};
/**
  @param ts_type distance;
  @param double time;
*/
struct bsf_snapshot {
    ts_type distance;
    double time;
} ;

/** Data structure for sorting the query.
  @param double value;
  @param      int  index;
 */
struct q_index {
    double value;
    int index;
} ;

static int cmp_pri(double next, double curr) {
    return (next > curr);
}



static double
get_pri(void *a) {
    return (double) ((struct query_result *) a)->distance;
}
static double
get_max_pri(void *a) {
    return (double) ((struct query_result *) a)->max_distance;
}

static void
set_pri(void *a, double pri) {
    ((struct query_result *) a)->distance = (float) pri;
}

static void
set_max_pri(void *a, double pri) {
    ((struct query_result *) a)->max_distance = (float) pri;
}


static size_t
get_pos(void *a) {
    return ((struct query_result *) a)->pqueue_position;
}


static void
set_pos(void *a, size_t pos) {
    ((struct query_result *) a)->pqueue_position = pos;
}


static double
get_pri2(void *a) {
    return (double) ((candidate *) a)->dist;
}
static void
set_pri2(void *a, double pri) {
    ((candidate *) a)->dist = (float) pri;
}


static size_t
get_pos2(void *a) {
    return ((candidate  *) a)->pqueue_position;
}


static void
set_pos2(void *a, size_t pos) {
    ((candidate *) a)->pqueue_position = pos;
}
//




// 并行查询中，每个工作线程私有的查询状态容器
typedef struct query_worker_data
{
    std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> * top_candidates;
    ts_type * kth_bsf;         // 	指向当前第 k 个最优（best-so-far） time series
    int id;                    
    querying_stats * stats;
    // float * localknn;
    std::vector<std::pair<float,unsigned int>> localknn;
    float bsf;
    unsigned short local_nprobes;
    bool end;
    unsigned short *flags;
    unsigned short curr_flag;

} worker_backpack__;












#endif //herculesHNSW_QUERYENGINE_H
