#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <ctime>
#include<time.h>
#include <cmath>
#include <queue>
#include <vector>
#include<set>
#include "QueryEngine.h"
#include "grasp_mu_bisect.h"
// #include <tbb/concurrent_unordered_map.h>
QueryEngine::~QueryEngine() {
    // if (pq) pqueue_free(pq);  // 释放pq发生段错误，不释放内存泄漏
    // if(results) free(results);
    // if(!parallel || nprobes <= 1) {
    //     if(flags) delete[] flags;
    // }
    // if(parallel and nprobes > 1){
    //     for(int i=1;i<nworker;i++){
    //         if(qwdata[i].stats) free(qwdata[i].stats);
    //         if(qwdata[i].top_candidates) delete qwdata[i].top_candidates;
    //         if(qwdata[i].localknn) free(qwdata[i].localknn);
    //         // if(qwdata[i].flags && qwdata[i].flags != flags) delete[] qwdata[i].flags;
    //     }
    //     free(qwdata);
    // }
    // if(flags) delete[] flags;

    if (this->flags) {
        delete[] this->flags;
        this->flags = nullptr;
    }

    // 释放 this->results
    if (this->results != nullptr) {
        for (int i = 0; i < this->query_dataset_size; i++) {
            if (this->results[i] != nullptr) {
                free(this->results[i]);          // 释放内层数组
                this->results[i] = nullptr;      // 避免悬空指针
            }
        }
        free(this->results);                     // 释放外层指针数组
        this->results = nullptr;
    }

    // 释放 this->learn_results
    if (this->learn_results != nullptr) {
        for (int i = 0; i < this->learn_dataset_size; i++) {
            if (this->learn_results[i] != nullptr) {
                free(this->learn_results[i]);    // 释放内层数组
                this->learn_results[i] = nullptr; // 避免悬空指针
            }
        }
        free(this->learn_results);               // 释放外层指针数组
        this->learn_results = nullptr;
    }
    

    if (this->pq) {
        pqueue_free(this->pq);  // 或者可能是 pqueue_destroy(this->pq)
        this->pq = nullptr;
    }


    if(parallel and nprobes > 1 && candidate_leaves) {
        // free(candidate_leaves);
        pqueue_free(candidate_leaves);  // 使用正确的清理函数，不是 free()
        candidate_leaves = nullptr;     
        

        if (qwdata) {
            // 释放每个工作线程的内部分配的内存
            for (int i = 1; i < nworker; i++) {
                // 释放统计数据
                if (qwdata[i].stats) {
                    free(qwdata[i].stats);
                    qwdata[i].stats = nullptr;
                }
                
                // 释放优先队列
                if (qwdata[i].top_candidates) {
                    delete qwdata[i].top_candidates;
                    qwdata[i].top_candidates = nullptr;
                }
                
                // 释放标志数组
                if (qwdata[i].flags) {
                    delete[] qwdata[i].flags;
                    qwdata[i].flags = nullptr;
                }
                
                // localknn 是 std::vector，会自动释放，不需要手动释放
            }
            
            // 释放 qwdata 数组本身（第79行分配的80字节）
            free(qwdata);
            qwdata = nullptr;
        }
        
        

    }

}


QueryEngine::QueryEngine(const char *query_filename, unsigned int query_dataset_size, 
                        const char *groundtruth_filename, unsigned int groundtruth_top_k, unsigned int groundtruth_dataset_size,
                        const char* learn_dataset, unsigned int learn_dataset_size, const char * learn_groundtruth_dataset,
                        const char* dataset,
                         Index *index, int ef, unsigned int nprobes, bool parallel,
                         unsigned int nworker, bool flatt ,unsigned int k, unsigned int ep,
                        const char *model_file, float zero_edge_pass_ratio) {

/*
* dataset: 为了验证无法剪枝引入的参数，其实是base dataset
*/
    
    this->dataset = dataset;
    this->query_filename = query_filename;
    this->query_dataset_size = query_dataset_size;

    this->groundtruth_filename=groundtruth_filename;
    this->groundtruth_dataset_size=groundtruth_dataset_size;
    this->groundtruth_top_k = groundtruth_top_k;

    this->learn_dataset=learn_dataset;
    this->learn_dataset_size=learn_dataset_size;
    this->learn_groundtruth_dataset = learn_groundtruth_dataset;
    this->model_file=model_file;
    this->zero_edge_pass_ratio = zero_edge_pass_ratio;

    this->index = index;
    this->index->ef = ef; 
    this->k=k;
    this->ep=ep;
    this->nprobes = nprobes; // 在查询过程中应当探查的候选节点数量
    this->parallel = parallel;
    this->nworker = nworker;
    // this->results = static_cast<float *>(malloc(sizeof(float) * k));
    this->candidate_leaves = nullptr;
    this->qwdata = nullptr;


    this->results = (int**)calloc(this->query_dataset_size, sizeof(int*));
    this->learn_results = (int**)calloc(this->learn_dataset_size, sizeof(int*));
    
    this->flags = new hnswlib::vl_type [Node::max_leaf_size];
    memset(this->flags, 0, sizeof(hnswlib::vl_type) * Node::max_leaf_size);
    this->curr_flag = 0;//max value of ushort is 65K => in total of search on all queries, we should not exceed 65k searchleaf

    this->pq = pqueue_init(Node::num_leaf_node, cmp_pri, get_pri, set_pri, get_pos, set_pos);


    //sometimes std::thread return 0 so, we use sysconf(LINUX)
    unsigned short local_nprobes = 0;

    if(parallel and nprobes >1){
        this->candidate_leaves =  pqueue_init(nprobes-1,
                                              cmp_pri, get_pri, set_pri, get_pos, set_pos);


        this->nworker = (std::thread::hardware_concurrency()==0)? sysconf(_SC_NPROCESSORS_ONLN) : std::thread::hardware_concurrency() -1;

        // this->nworker = 1;
        if(this->nworker>nprobes-1)this->nworker = nprobes-1;
        this->qwdata = static_cast<worker_backpack__ *>(malloc(sizeof(worker_backpack__) * this->nworker));

        local_nprobes = (nprobes-1) / this->nworker;
        for(int i=1;i<this->nworker;i++){
            qwdata[i].stats = static_cast<querying_stats *>(malloc(sizeof(querying_stats)));
            qwdata[i].top_candidates = new std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>>();
            qwdata[i].local_nprobes = (local_nprobes);               //  每个线程负责查询的叶子节点数目
            qwdata[i].localknn.reserve(k);

            qwdata[i].flags = new hnswlib::vl_type [Node::max_leaf_size];
            memset(qwdata[i].flags, 0, sizeof(hnswlib::vl_type) * Node::max_leaf_size);
            qwdata[i].curr_flag = 0;

        }
       // qwdata[0].localknn = static_cast<float *>(malloc(k*sizeof(float)));
        qwdata[0].local_nprobes += (nprobes-1)%this->nworker;
        qwdata[0].stats = &stats;                                // global  stats
        // qwdata[0].top_candidates = &top_candidates;             //  global   top_candidates
        qwdata[0].top_candidates = new std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>>();

        omp_set_dynamic(0);
        omp_set_num_threads(this->nworker);

    }
    this->setEF(this->index->first_node,ef);
    if(parallel){
        std::cout << "[QUERYING PARAM]  EF : " <<this->index->ef
                  <<"| NPROBES : " <<nprobes
                  <<"| PARALLEL : "<<this->parallel
                  <<"| NWORKERS : " <<this->nworker
                  <<"| Nprobes/Worker :"<<local_nprobes
                  <<std::endl;
    }
    else{
        std::cout << "[QUERYING PARAM]  EF : " <<this->index->ef
                  <<"| NPROBES : " <<nprobes
                  <<"| PARALLEL : "<<0
                  <<std::endl;
    }
}


void QueryEngine::setEF(Node * node, int ef){
    if(node == nullptr) {
        std::cerr << "Warning: setEF called with null node pointer" << std::endl;
        return;
    }
    
    if(node->is_leaf) {
        if(node->leafgraph != nullptr) {
            node->leafgraph->setEf(ef);
        } else {
            std::cerr << "Warning: leaf node has null leafgraph pointer" << std::endl;
        }
    } else {
        if(node->left_child != nullptr) {
            setEF(node->left_child, ef);
        }
        if(node->right_child != nullptr) {
            setEF(node->right_child, ef);
        }
    }
}
void QueryEngine::queryBinaryFile(unsigned int k, int mode, bool search_withWeight, float thres_probability, float μ, float T) {

    if(this->groundtruth_dataset_size != this->query_dataset_size ){
        throw std::runtime_error("groundtruth dataset size != query dataset size");
        exit(-1);
    }

    cout << "[Querying] " << this->query_filename << endl;

    /*  open query_file and groundtruth_file */
    this->groundtruth_file = fopen(this->groundtruth_filename, "rb");
    if(this->groundtruth_file==nullptr){
        fprintf(stderr, "Groundtruth file %s not found!\n", this->groundtruth_filename);
        exit(-1);
    }

    this->query_file = fopen(this->query_filename, "rb");
    if (this->query_file  == nullptr) {
        fprintf(stderr, "Queries file %s not found!\n", this->query_filename);
        exit(-1);
    }

    fseek(this->query_file, 0L, SEEK_END);
    file_position_type sz = (file_position_type) ftell(this->query_file);
    fseek(this->query_file, 0L, SEEK_SET);
    this->total_records = sz / (this->index->index_setting->timeseries_size * sizeof(ts_type));


    fseek(this->query_file, 0L, SEEK_SET);
    unsigned int offset = 0;

    if (this->total_records < this->query_dataset_size) {
        fprintf(stderr, "File %s has only %llu records!\n", query_filename, total_records);
        exit(-1);
    }
    cout << this->total_records << " records in the query file" << endl;


    unsigned int q_loaded = 0;
    unsigned int ts_length = this->index->index_setting->timeseries_size; // dimension

    
    /* // open model
    try{
        Module model = torch::jit::load(model_file);
    }catch (const c10::Error& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return -1;
    } */

    /*  malloc memory for query and groundtruth */
    ts_type *query_ts = static_cast<ts_type *>(malloc_search( sizeof(ts_type) * ts_length));
    int *groundtruth_id = static_cast<int *>(malloc_search( sizeof(int) * groundtruth_top_k));
    

    // Record start time
    auto start = now();
    while(q_loaded < this->query_dataset_size){

        /* load query */
        fread(query_ts, sizeof(ts_type), ts_length, this->query_file); 

        /* load groundtruth */
        fread(groundtruth_id, sizeof(int), groundtruth_top_k, this->groundtruth_file);    

        searchNpLeafParallel(query_ts, groundtruth_id, k, nprobes, q_loaded, search_withWeight, thres_probability, μ, T);

        q_loaded++;

    }

    free(query_ts); 
    free(groundtruth_id);
    this->closeFile();

    index->time_stats->querying_time = getElapsedTime(start);

}

void QueryEngine::searchNpLeafParallel(ts_type *query_ts, int* groundtruth_id, unsigned int k, unsigned int nprobes, unsigned int query_index, 
                                      bool search_withWeight, float thres_probability, float μ, float T) {

    stats.reset();
    Time start = now();    
    ts_type kth_bsf = FLT_MAX;  // global  kth_bsf


    this->results[query_index]=(int*)calloc(k, sizeof(int));
    if (this->results[query_index] == nullptr) {
        fprintf(stderr, "Memory allocation failed for this->results[%d]\n", query_index);
        exit(-1);
    }

    // step0: route query_ts from root to the leaf node it belongs to
    Node *App_node = this->index->first_node; 
    ts_type App_bsf = FLT_MAX;
    if (App_node == nullptr) throw std::runtime_error("Error : First node == nullptr!");
    while (!App_node->is_leaf) { 
        if (App_node->node_split_policy_route_to_left(query_ts)) {
            App_node = App_node->left_child;
        } else {
            App_node = App_node->right_child;
        }
    }

    // // all_leaf_nodes_dis: 所有叶子节点的距离
    //     float top1_distance = 0;
    // {
    //     // query的top1结果下标为groundtruth_id[0]，需要根据下标到dataset中找到对应的top1向量，并计算其距离
    
    //     // 打开dataset文件
    //     FILE *dataset_file = fopen(this->dataset, "rb");
    //     if (dataset_file == nullptr) {
    //         fprintf(stderr, "Dataset file %s not found!\n", this->dataset);
    //         exit(-1);
    //     }
    
    //     // 根据groundtruth_id[0]找到对应的向量
    //     fseek(dataset_file, groundtruth_id[0] * this->index->index_setting->timeseries_size * sizeof(ts_type), SEEK_SET);
    //     ts_type *top1_vector = static_cast<ts_type *>(malloc_search(sizeof(ts_type) * this->index->index_setting->timeseries_size));
    //     fread(top1_vector, sizeof(ts_type), this->index->index_setting->timeseries_size, dataset_file);
    
    //     // 计算top1向量与query_ts的欧几里得距离
    //     for (int i = 0; i < this->index->index_setting->timeseries_size; i++) {
    //         top1_distance += (query_ts[i] - top1_vector[i]) * (query_ts[i] - top1_vector[i]);
    //     }
    //     cout<<"top1_distance:"<<top1_distance<<endl;
    
    //     // 释放top1向量
    //     free(top1_vector);
    // }
    // float *all_leaf_nodes_dis = (float*)calloc(Node::num_leaf_node, sizeof(float));
    // all_leaf_nodes_dis[0] = App_node->calculate_node_min_distance(this->index, query_ts, stats);
    // int leaf_node_count = 1;

    // 搜索top-k作为参考
    searchGraphLeaf(App_node,query_ts, k,
                    top_candidates, App_bsf, stats, flags,
                    curr_flag, search_withWeight, thres_probability, μ, T);

    cout<<"************************************************"<<endl;
    cout<<"query_index:"<<query_index<<endl;
    float recall1 = calculateRecall(top_candidates, groundtruth_id, k, groundtruth_top_k);
    cout<<"recall1:"<<recall1<<endl;

    nprobes--;
    if (nprobes == 0) {
        while (top_candidates.size() > k)top_candidates.pop();

        int m=0;
        while (top_candidates.size() > 0) {

            // results[top_candidates.size() - 1] = top_candidates.top().first;
            // topkID.push_back(top_candidates.top());
            this->results[query_index][m++]=top_candidates.top().second;
            top_candidates.pop();
        }
        double time = getElapsedTime(start);
        // printKNN(topkID, k, time, visited);

    }
    else{

    // step2: get candidate leaf
    
        auto *root_pq_item = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
        root_pq_item->node = this->index->first_node;
        root_pq_item->distance = this->index->first_node->calculate_node_min_distance(this->index, query_ts, stats);
        pqueue_insert(pq, root_pq_item); 

        struct query_result *n;
        ts_type child_distance;
        ts_type bsf = FLT_MAX;


        query_result * candidates =  static_cast<query_result *>(calloc(Node::num_leaf_node+1, sizeof(struct query_result))); // 数组，距离
        unsigned int candidates_count = 0;
        unsigned int computation_count = 0;
        int pos;
        kth_bsf = top_candidates.top().first; // current top-k_th timeseries distance
        while ((n = static_cast<query_result *>(pqueue_pop(pq)))) {
            if (n->distance > kth_bsf) {//getting through two pruning process is tricky...
                break;
            }
            if (n->node->is_leaf) 
            {
                // all_leaf_nodes_dis[leaf_node_count++] = n->distance; // 记录所有叶子节点的距离,用于查看叶子剪枝效率

                pos = candidates_count - 1; 

                if (pos >= 0)
                    while (pos >= 0 and n->distance < candidates[pos].distance) {     /* 这种插入方法使得 candidates 是有序的，按照每个元素的距离递增排序  */
                        candidates[pos + 1].node = candidates[pos].node;
                        candidates[pos + 1].distance = candidates[pos].distance;
                        pos--;
                    }
                candidates[pos + 1].node = n->node;
                candidates[pos + 1].distance = n->distance;
                candidates_count++;
            } else            
            {
                // check node->left_child
                computation_count++;
                child_distance = n->node->left_child->calculate_node_min_distance(this->index, query_ts, stats);
                if ((child_distance < kth_bsf) && (n->node->left_child != App_node)) //add epsilon
                {
                    auto *mindist_result_left = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
                    mindist_result_left->node = n->node->left_child;
                    mindist_result_left->distance = child_distance;
                    pqueue_insert(pq, mindist_result_left);
                }

                // check node->right_child
                computation_count++;
                child_distance = n->node->right_child->calculate_node_min_distance(this->index, query_ts, stats);
                if ((child_distance < kth_bsf) && (n->node->right_child != App_node)) 
                {
                    auto *mindist_result_right = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
                    mindist_result_right->node = n->node->right_child;
                    mindist_result_right->distance = child_distance;
                    pqueue_insert(pq, mindist_result_right);
                }
            }

            free(n);
        }
        stats.num_candidates = candidates_count;

/*         将所有叶子节点的距离写入文件
        {
            //将所有叶子节点的距离进行排序
            sort(all_leaf_nodes_dis+1, all_leaf_nodes_dis + leaf_node_count);
            std::string dis_dataset_dir = this->index->index_path;
            std::string leaf_nodes_dis_file = dis_dataset_dir + "/leaf_nodes_dis.txt";
         
            std::ofstream outfile(leaf_nodes_dis_file, std::ios::app);
            if (!outfile) {
                std::cerr << "Cannot open file: " << leaf_nodes_dis_file << std::endl;
            }
            outfile<<top1_distance<<" ";
            for (int i = 1; i < leaf_node_count; i++) {
                outfile << all_leaf_nodes_dis[i] << " ";
            }
            outfile<<endl;
            outfile.close();
        }
 */
        // 将搜索的叶子节点的数量和计算的次数写入文件
        /* char num_candidates_file[256];
        char num_computation_count[256];
        sprintf(num_candidates_file, "%s/num_candidates_%d.txt", this->index->index_path, k);
        sprintf(num_computation_count, "%s/computation_count_%d.txt", this->index->index_path, k);
    
         // 打开文件，追加模式（ios::app）
         std::ofstream outfile(num_candidates_file, std::ios::app);
         if (!outfile) {
             std::cerr << "无法打开文件 " << num_candidates_file << std::endl;
         }
    
         std::ofstream outfile2(num_computation_count, std::ios::app);
         if (!outfile2) {
             std::cerr << "无法打开文件 " << num_computation_count << std::endl;
         }
    
         // 写入内容
         outfile << candidates_count << std::endl;
         outfile2 << computation_count << std::endl;
        cout<<"stats.num_candidates:" << stats.num_candidates<<endl;
        cout<<"computation_count:"<<computation_count<<endl; */

    // step3： parallel search the graph of the leaf node

        // 3.1 每个 qwdata 工作单元负责一个线程的数据处理任务
        if (parallel and nprobes > 1) { 
            for (int i = 1; i < nworker; i++) {
                qwdata[i].id = i;
                qwdata[i].kth_bsf = &kth_bsf;
                qwdata[i].stats->reset();
                qwdata[i].bsf = FLT_MAX;
            }
            qwdata[0].kth_bsf = &kth_bsf;   // global kth_bsf
            qwdata[0].bsf = FLT_MAX;
            qwdata[0].id=0;
            qwdata[0].flags = flags;
            qwdata[0].curr_flag = curr_flag;

            copypq(qwdata, top_candidates); 

            pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER; 

            query_result node;
            query_worker_data *worker;

        // 3.2 开始并行搜索
            {
 #pragma omp parallel num_threads(nworker) private(node, bsf, worker) shared(qwdata, candidates_count, candidates,  query_ts, k)
                {
                    bsf = FLT_MAX;
                    worker = qwdata+omp_get_thread_num(); // 根据当前线程的编号（omp_get_thread_num()）选择相应的 worker 数据结构

                    // 3.2.1 每个线程处理一个循环迭代
#pragma omp for schedule(static, 1) 
                    // for (int i = 0; i < std::min(candidates_count,nprobes); i++) {     
                    for (int i = 0; i < candidates_count; i++) {     

                        node = candidates[i];
                        pthread_rwlock_rdlock(&lock_bsf);
                        bsf = *worker->kth_bsf; // 每个线程的 worker 结构体中存储了对共享的 bsf 的引用
                        pthread_rwlock_unlock(&lock_bsf);
                        worker->stats->num_leaf_checked++;
                        //                    worker->checked_leaf.push(node.node->id);
                        if (node.distance <= worker->bsf) { // 只有满足一定条件，才搜索node
                            worker->stats->num_leaf_searched++;

                            searchGraphLeaf(node.node, query_ts,  k, *(worker->top_candidates), worker->bsf, 
                                             *(worker->stats), worker->flags, worker->curr_flag, search_withWeight, thres_probability, μ, T);

                            if (worker->top_candidates->top().first < bsf) {
                                pthread_rwlock_wrlock(&lock_bsf);
                                *(worker->kth_bsf) = worker->top_candidates->top().first; // 更新global kth_bsf
                                pthread_rwlock_unlock(&lock_bsf);
                            }

                        }
                    }
                }
            }

  
    // step4： 合并中间结果
            for(int i=0; i<nworker; i++){
                if(qwdata[i].stats->num_leaf_searched==0){
                    while(qwdata[i].top_candidates->size()>0){
                        qwdata[i].top_candidates->pop();
                    }
                    continue;
                }
                while(qwdata[i].top_candidates->size()!=0){
                    if(qwdata[i].top_candidates->top().second==UINT32_MAX){
                        qwdata[i].top_candidates->pop();
                        continue;
                    }
                    else if(qwdata[i].top_candidates->top().first < top_candidates.top().first) {
                        top_candidates.emplace(qwdata[i].top_candidates->top());   

                        while(top_candidates.size()>k){
                            top_candidates.pop();
                        }
                    } 
                    qwdata[i].top_candidates->pop();

                }
            }
        } // end if (parallel and nprobes > 1) 

        float recall2 = calculateRecall(top_candidates, groundtruth_id, k, groundtruth_top_k);
        cout<<"recall2:"<<recall2<<endl;
        // outfile.close();
        // outfile2.close();


        /* 统计topkID */
        int m=0;
        while (top_candidates.size() > 0) {

            // results[top_candidates.size() - 1] = top_candidates.top().first;
            // topkID.push_back(top_candidates.top());
            this->results[query_index][m]=top_candidates.top().second;
            m++;
            top_candidates.pop();
        }

        double time = getElapsedTime(start);


        /* // 4.3 update global stats
        if (parallel and nprobes > 1) {
            for (int i = 1; i < nworker; i++) {

               // cout << "worker "<<i<<": visited "<<qwdata[i].stats->num_leaf_searched<<endl;
                stats.num_knn_alters += qwdata[i].stats->num_knn_alters;
                stats.num_leaf_checked += qwdata[i].stats->num_leaf_checked;
                stats.num_leaf_searched += qwdata[i].stats->num_leaf_searched;
                stats.distance_computations_hrl += qwdata[i].stats->distance_computations_hrl;
                stats.distance_computations_bsl += qwdata[i].stats->distance_computations_bsl;

            }
        } */

        // 4.4 输出结果数组 results
        // printKNN(topkID, k, time, visited, parallel && nprobes > 1);
        // printKNN(topkID, k, time, visited);


        // 4.5 Free the nodes that were not popped.
        while ((n = static_cast<query_result *>(pqueue_pop(pq)))) free(n);
        if (parallel and nprobes > 1) {
            while ((n = static_cast<query_result *>(pqueue_pop(candidate_leaves)))) free(n);
            qwdata[0].flags = nullptr;
        }
        pq->size = 1;
        free(candidates);
    }

}


// void QueryEngine::searchPredictedNpLeafParallel(ts_type *query_ts, int* groundtruth_id, unsigned int k, unsigned int nprobes, unsigned int query_index) {

//     stats.reset();

//     Time start = now();

//     ts_type kth_bsf = FLT_MAX;  // global  kth_bsf

//     this->results[query_index]=(int*)calloc(k, sizeof(int));
//     if (this->results[query_index] == nullptr) {
//         fprintf(stderr, "Memory allocation failed for this->results[%d]\n", query_index);
//         exit(-1);
//     }


//     // step0: route query_ts from root to the leaf node it belongs to
//     Node *App_node = this->index->first_node; 
//     ts_type App_bsf = FLT_MAX;
//     if (App_node == nullptr) throw std::runtime_error("Error : First node == nullptr!");
//     while (!App_node->is_leaf) { 
//         if (App_node->node_split_policy_route_to_left(query_ts)) {
//             App_node = App_node->left_child;
//         } else {
//             App_node = App_node->right_child;
//         }
//     }

//     // step1: search App_node to get current top-k candidate and save in top_candidates
//     /* 
//      * flags :
//      * curr_flag :
//      */
//     searchGraphLeaf(App_node,query_ts, k,
//                     top_candidates, App_bsf, stats, flags,
//                     curr_flag, false);

//     cout<<"************************************************"<<endl;


//     nprobes--;
//     if (nprobes == 0) {  // one leaf node 
//         int m=0;
//         while (top_candidates.size() > 0) {
//             this->results[query_index][m++]=top_candidates.top().second;
//             top_candidates.pop();
//         }
//         double time = getElapsedTime(start);
//         // printKNN(topkID, k, time, visited);
//     }
//     else{

//     // step2: get candidate leaf
    
//          // else.1 insert root_pq_item into pq
//         auto *root_pq_item = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
//         root_pq_item->node = this->index->first_node;
//         root_pq_item->distance = this->index->first_node->calculate_node_min_distance(this->index, query_ts, stats);
//         pqueue_insert(pq, root_pq_item); 

//         struct query_result *n;
//         ts_type child_distance;
//         ts_type bsf = FLT_MAX;


//         // else.2 pruning internal nodes and leaf nodes
//         query_result * candidates =  static_cast<query_result *>(calloc(Node::num_leaf_node+1, sizeof(struct query_result))); // 数组，距离
//         unsigned int candidates_count = 0;
//         int pos;
//         while ((n = static_cast<query_result *>(pqueue_pop(pq)))) {
//             if (n->node->is_leaf) // n is a leaf
//             {
//                 candidates[candidates_count].node = n->node;
//                 candidates[candidates_count].distance = n->distance;
//                 candidates_count++;
//             } else                // n is a internal leaf
//             {
//                 // check node->left_child
//                 child_distance = n->node->left_child->calculate_node_min_distance(this->index, query_ts, stats);
//                 if (n->node->left_child != App_node) //add epsilon
//                 {
//                     auto *mindist_result_left = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
//                     mindist_result_left->node = n->node->left_child;
//                     mindist_result_left->distance = child_distance;
//                     pqueue_insert(pq, mindist_result_left);
//                 }

//                 // check node->right_child
//                 child_distance = n->node->right_child->calculate_node_min_distance(this->index, query_ts, stats);
//                 if (n->node->right_child != App_node) //add epsilon
//                 {
//                     auto *mindist_result_right = static_cast<query_result *>(malloc_search(sizeof(struct query_result)));
//                     mindist_result_right->node = n->node->right_child;
//                     mindist_result_right->distance = child_distance;
//                     pqueue_insert(pq, mindist_result_right);
//                 }
//             }

//             free(n);
//         } 
//         stats.num_candidates = candidates_count;


    

//         std::vector<std::pair<Node*, float>> predicted_nodes = model.predict_nodes(query_ts, nprobes); // 假设返回的叶子节点已经按照概率排序

//         for (const auto& pred : predicted_nodes) {
//             Node* node = pred.first;
//             float score = pred.second;
        
//             // 仅处理叶节点，且不是 App_node
//             if ( node != App_node) {
//                 // 按分数升序插入 candidates 数组
//                 int pos = candidates_count - 1;
//                 while (pos >= 0 && score < candidates[pos].distance) {
//                     candidates[pos + 1].node = candidates[pos].node;
//                     candidates[pos + 1].distance = candidates[pos].distance;
//                     pos--;
//                 }
//                 candidates[pos + 1].node = node;
//                 candidates[pos + 1].distance = score; // 使用神经网络预测的分数
//                 candidates_count++;
//             }
//         }
//         stats.num_candidates = candidates_count;
        

//     // step3： parallel search the graph of the leaf node

//         // 3.1 每个 qwdata 工作单元负责一个线程的数据处理任务
//         if (parallel and nprobes > 1) { 
//             for (int i = 1; i < nworker; i++) {
//                 qwdata[i].id = i;
//                 qwdata[i].kth_bsf = &kth_bsf;
//                 qwdata[i].stats->reset();
//                 qwdata[i].bsf = FLT_MAX;
//             }
//             qwdata[0].kth_bsf = &kth_bsf;   // global kth_bsf
//             qwdata[0].bsf = FLT_MAX;
//             qwdata[0].id=0;
//             qwdata[0].flags = flags;
//             qwdata[0].curr_flag = curr_flag;

//             copypq(qwdata, top_candidates); 

//             pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER; 

//             query_result node;
//             query_worker_data *worker;

//         // 3.2 开始并行搜索
//             {
//                 /* 
//                  * num_threads(nworker)： 要启动的线程数nworker
//                  * private(node, bsf, worker)：每个线程都有自己独立的 node、bsf 和 worker 变量
//                  * shared(qwdata, candidates_count, candidates,  query_ts, k)：所有线程的共享变量
//                  */
//                 #pragma omp parallel num_threads(nworker) private(node, bsf, worker) shared(qwdata, candidates_count, candidates,  query_ts, k)
//                 {
//                     bsf = FLT_MAX;
//                     worker = qwdata+omp_get_thread_num(); // 根据当前线程的编号（omp_get_thread_num()）选择相应的 worker 数据结构

//                     // 3.2.1 每个线程处理一个循环迭代
//                     #pragma omp for schedule(static, 1) 
//                     for (int i = 0; i < std::min(candidates_count,nprobes); i++) {     

//                         node = candidates[i];
//                         pthread_rwlock_rdlock(&lock_bsf);
//                         bsf = *worker->kth_bsf; // 每个线程的 worker 结构体中存储了对共享的 bsf 的引用
//                         pthread_rwlock_unlock(&lock_bsf);
//                         worker->stats->num_leaf_checked++;
//                         //                    worker->checked_leaf.push(node.node->id);
//                         if (node.distance <= worker->bsf) { // 只有满足一定条件，才搜索node
//                             worker->stats->num_leaf_searched++;

//                             searchGraphLeaf(node.node, query_ts,  k, *(worker->top_candidates), worker->bsf, 
//                                              *(worker->stats), worker->flags, worker->curr_flag, true);

//                             if (worker->top_candidates->top().first < bsf) {
//                                 pthread_rwlock_wrlock(&lock_bsf);
//                                 *(worker->kth_bsf) = worker->top_candidates->top().first; // 更新global kth_bsf
//                                 pthread_rwlock_unlock(&lock_bsf);
//                             }

//                         }
//                     }
//                 }
//             }
//     // step4： 合并中间结果
//         // 4.1 根据每个qwdata[i].localknn[j]  更新 top_candidates
//             for(int i=1; i<nworker; i++){
//                 if(qwdata[i].stats->num_leaf_searched==0){
//                     while(qwdata[i].top_candidates->size()>0){
//                         qwdata[i].top_candidates->pop();
//                     }
//                     continue;
//                 }
//                 while(qwdata[i].top_candidates->size()!=0){
//                     if(qwdata[i].top_candidates->top().first < top_candidates.top().first) {
//                         top_candidates.emplace(qwdata[i].top_candidates->top());   
//                         // top_candidates.pop();
//                         while(top_candidates.size()>k){
//                             top_candidates.pop();
//                         }
//                     } 
//                     qwdata[i].top_candidates->pop();
//                 }
//             }
//         } // end if (parallel and nprobes > 1) 


                
//         float recall2 = calculateRecall(top_candidates, groundtruth_id, k, groundtruth_top_k);
//         cout<<"recall2:"<<recall2<<endl;


//         /* 统计topkID */
//         int m=0;
//         while (top_candidates.size() > 0) {
//             this->results[query_index][m]=top_candidates.top().second;
//             m++;
//             top_candidates.pop();
//         }

//         double time = getElapsedTime(start);


//         // 4.3 update global stats
//         if (parallel and nprobes > 1) {
//             for (int i = 1; i < nworker; i++) {

//                // cout << "worker "<<i<<": visited "<<qwdata[i].stats->num_leaf_searched<<endl;
//                 stats.num_knn_alters += qwdata[i].stats->num_knn_alters;
//                 stats.num_leaf_checked += qwdata[i].stats->num_leaf_checked;
//                 stats.num_leaf_searched += qwdata[i].stats->num_leaf_searched;
//                 stats.distance_computations_hrl += qwdata[i].stats->distance_computations_hrl;
//                 stats.distance_computations_bsl += qwdata[i].stats->distance_computations_bsl;

//             }
//         }

//         // 4.4 输出结果数组 results
//         // printKNN(topkID, k, time, visited, parallel && nprobes > 1);
//         // printKNN(topkID, k, time, visited);


//         /* // 4.5 Free the nodes that were not popped.
//         while ((n = static_cast<query_result *>(pqueue_pop(pq)))) free(n);
//         if (parallel and nprobes > 1) {
//             while ((n = static_cast<query_result *>(pqueue_pop(candidate_leaves)))) free(n);
//             qwdata[0].flags = nullptr;
//         } */
//         pq->size = 1;
//         free(candidates);
//     }

// }


void QueryEngine::TrainWeightByLearnDataset(IterRefinement_epoch ep, unsigned int k, std::vector<std::set<Node*>> candidate_leaf_node) {


    if(ep==0)
        return;

    // preparation
    this->candidate_leaf_node=candidate_leaf_node;
    FILE *lfile=fopen(this->learn_dataset, "rb");
    if(lfile==nullptr){
        throw std::runtime_error("Open learn dataset failed! ");
    }

    fseek(lfile, 0L, SEEK_END);
    file_position_type sz = (file_position_type) ftell(lfile);
    fseek(lfile, 0L, SEEK_SET);
    unsigned int total_learn_records = sz / (this->index->index_setting->timeseries_size * sizeof(ts_type));
    fseek(lfile, 0L, SEEK_SET);

    if(total_learn_records < this->learn_dataset_size){
        fprintf(stderr,"Actual total records %d < learn_dataset_size %d\n",total_learn_records,this->learn_dataset_size);
    }


    this->learn_groundtruth_file = fopen(this->learn_groundtruth_dataset, "rb");
    if(this->learn_groundtruth_file==nullptr){
        fprintf(stderr, "Groundtruth file %s not found!\n", this->learn_groundtruth_dataset);
    }
    unsigned int learn_loaded = 0;
    unsigned int ts_length = this->index->index_setting->timeseries_size; // dimension


    ts_type *learn_ts = static_cast<ts_type *>(malloc_search( sizeof(ts_type) * ts_length));
    int *learn_groundtruth_id = static_cast<int *>(malloc_search( sizeof(int) * groundtruth_top_k));

    unsigned int ep_index=0;
    // first epoch
    cout<<"first epoch begining"<<endl;
    while(learn_loaded < learn_dataset_size){

        fread(learn_ts, sizeof(ts_type), ts_length, lfile);
        fread(learn_groundtruth_id, sizeof(int), groundtruth_top_k,  this->learn_groundtruth_file);
        TrainWeightinNpLeafParallel(learn_ts, learn_groundtruth_id, k, nprobes, learn_loaded, ep_index, this->candidate_leaf_node[learn_loaded]);
        learn_loaded++;

    }
    cout<<"first epoch ended"<<endl;

    // remaining epoch
    cout<<"remaining epoch begining"<<endl;
    for (ep_index = 1; ep_index < ep; ep_index++)
    {
        fseek(lfile, 0L, SEEK_SET);
        fseek(this->learn_groundtruth_file, 0L, SEEK_SET);
        learn_loaded=0;
        while(learn_loaded < learn_dataset_size){

            fread(learn_ts, sizeof(ts_type), ts_length, lfile);
            fread(learn_groundtruth_id, sizeof(int), groundtruth_top_k,  this->learn_groundtruth_file);

    
            TrainWeightinNpLeafParallel(learn_ts, learn_groundtruth_id, k, nprobes, learn_loaded, ep_index, this->candidate_leaf_node[learn_loaded]);
            learn_loaded++;

        }
    }
    cout<<"remaining epoch ended"<<endl;


    // 获取所有叶子节点列表,以便输出所有叶子节点的边权重到文件
    Node* root_node = this->index->first_node;
    Node** leaves = static_cast<Node**>(malloc_search( sizeof(Node*) * Node::num_leaf_node));
    int count = 0;
    root_node->getLeaves(leaves, count);

    // 输出所有叶子节点的边权重到文件
    for(int j=0; j<count; j++){
        char* leaf_path_c = leaves[j]->getLeafGraphFullFileName(this->index);
        std::string edge_weight_file = std::string(leaf_path_c) + "_edge_weight.txt";
        free(leaf_path_c);
        if(leaves[j]->leafgraph == nullptr) {
            std::cerr << "Warning: leaf " << j << " has null leafgraph, skipping weight output..." << std::endl;
            continue;
        }
        auto g = leaves[j]->leafgraph;
        // 若能访问 cur_element_count 用它；否则用 max_elements_ 并判空邻接表
        size_t N = g->max_elements_;
        
        for (size_t u = 0; u < N; u++) {
            int* data = (int*)g->get_linklist0(u);
            if (data == nullptr) continue;
            size_t deg = g->getListCount((unsigned int*)data);
            for (size_t k = 1; k <= deg; k++) {
                int v = *(data + k);
                float w = g->edgeWeightList_[u][v].weight;  // 直接二维索引
                // 写出 (u, v, w)
                std::ofstream outfile(edge_weight_file, std::ios::app);
                outfile << u << " " << v << " " << w << std::endl;
                outfile.close();
            }
        }
    } 
    free(leaves);

    fclose(lfile);
    fclose(learn_groundtruth_file);
    free(learn_ts);
    free(learn_groundtruth_id);
}

void QueryEngine::queryWithWeight(unsigned int k, int mode, bool search_withWeight, float thres_probability, float μ, float T, std::vector<std::set<Node *>> candidate_leaf_node){

    if(this->groundtruth_dataset_size != this->query_dataset_size ){
        throw std::runtime_error("groundtruth dataset size != query dataset size");
        exit(-1);
    }


    /*  open query_file and groundtruth_file */
    this->groundtruth_file = fopen(this->groundtruth_filename, "rb");
    if(this->groundtruth_file==nullptr){
        fprintf(stderr, "Groundtruth file %s not found!\n", this->groundtruth_filename);
        exit(-1);
    }

    cout<<"************************************************"<<endl;
    cout << "[Querying] " << this->query_filename << endl;
    this->query_file = fopen(this->query_filename, "rb");
    if (this->query_file  == nullptr) {
        fprintf(stderr, "Queries file %s not found!\n", this->query_filename);
        exit(-1);
    }

    fseek(this->query_file, 0L, SEEK_END);
    file_position_type sz = (file_position_type) ftell(this->query_file);
    fseek(this->query_file, 0L, SEEK_SET);
    this->total_records = sz / (this->index->index_setting->timeseries_size * sizeof(ts_type));


    fseek(this->query_file, 0L, SEEK_SET);
    unsigned int offset = 0;

    if (this->total_records < this->query_dataset_size) {
        fprintf(stderr, "File %s has only %llu records!\n", query_filename, total_records);
        exit(-1);
    }
    cout << this->total_records << " records in the query file" << endl;


    unsigned int q_loaded = 0;
    unsigned int ts_length = this->index->index_setting->timeseries_size; // dimension

    

    /*  malloc memory for query and groundtruth */
    ts_type *query_ts = static_cast<ts_type *>(malloc_search( sizeof(ts_type) * ts_length));
    int *groundtruth_id = static_cast<int *>(malloc_search( sizeof(int) * groundtruth_top_k));
    

    // Record start time
    auto start = now();
    while(q_loaded < this->query_dataset_size){

        /* load query */
        fread(query_ts, sizeof(ts_type), ts_length, this->query_file); 

        /* load groundtruth */
        fread(groundtruth_id, sizeof(int), groundtruth_top_k, this->groundtruth_file);    

        searchWithWeightinNpLeafParallel(query_ts, groundtruth_id, k, nprobes, q_loaded, 
                                        search_withWeight, thres_probability, μ, T, candidate_leaf_node[q_loaded]);

        q_loaded++;

    }

    free(query_ts); 
    free(groundtruth_id);
    this->closeFile();

    index->time_stats->querying_time = getElapsedTime(start);

}


void QueryEngine::closeFile(){
    if (fclose(this->query_file)) {
        fprintf(stderr, "Error: Could not close the query filename %s", query_filename);
        exit(-1);
    }

    if (fclose(this->groundtruth_file)) {
        fprintf(stderr, "Error: Could not close the groundtruth filename %s", query_filename);
        exit(-1);
    }

    /* if (fclose(this->learn_groundtruth_file)) {
        fprintf(stderr, "Error: Could not close the groundtruth filename %s\n", query_filename);
        exit(-1);
    } */
    
    printf( "[Query && groundtruth  && learn_groundtruth File Closed] %s \n", query_filename);
}


/* 计算单个 query 的 recall*/
float QueryEngine::calculateRecall(std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>>top_candidates, 
                      int* groundtruth_id, unsigned int k, unsigned int groundtruth_top_k) {

    if (groundtruth_top_k < k) {
        throw std::runtime_error("Groundtruth top-k must be greater than or equal to k.");
    }

    // 将 groundtruth_id 中前 k 个节点存入一个 set，用于快速查找
    std::unordered_set<unsigned int> groundtruth_set;
    for (unsigned int i = 0; i < k; i++) {
        groundtruth_set.insert(groundtruth_id[i]);
    }

    // 计算 top_candidates 中前 k 个结果与 groundtruth_id 中前 k 个相关节点的交集
    unsigned int relevant_count = 0;
    for (unsigned int i = 0; i < k; i++) {
        if (top_candidates.empty()) break;

        unsigned int candidate_id = top_candidates.top().second;
        top_candidates.pop();  // 弹出当前节点

        // 检查当前候选节点是否在 groundtruth_set 中
        if (groundtruth_set.find(candidate_id) != groundtruth_set.end()) {
            relevant_count++;  // 如果在，说明这个节点是相关的
        }
    }

    // 计算 recall
    float recall = static_cast<float>(relevant_count) / k;

    return recall;
}


double QueryEngine::calculateAverageRecall(){
    int intersection_size =0;
    this->groundtruth_file = fopen(this->groundtruth_filename, "rb");
    if(this->groundtruth_file==nullptr){
        fprintf(stderr, "Groundtruth file %s not found!\n", this->groundtruth_filename);
        exit(-1);
    }
    int *groundtruth_id = static_cast<int *>(malloc_search( sizeof(int) * groundtruth_top_k));

    for (size_t i = 0; i < this->query_dataset_size; i++)
    {
        fread(groundtruth_id, sizeof(int), groundtruth_top_k, this->groundtruth_file);
        for (size_t j = 0; j < this->k; j++)
        {
            for(int n=0; n< this->k; n++){
                if(this->results[i][j]==groundtruth_id[n]){
                    intersection_size++;
                    break;
                }
            }
            
        }
        
    }

    double averageRecall=intersection_size*1.0/(k*this->query_dataset_size);
    
    free(groundtruth_id);
    fclose(this->groundtruth_file);

    return  averageRecall;
}


void QueryEngine::searchflat(Node * node, unsigned int entrypoint, const void *data_point, size_t beamwidth,size_t k,
                std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> & top_candidates,
                float & bsf,querying_stats & stats, unsigned short *threadvisits, unsigned short & round_visit) {
    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: searchflat called on leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    auto g = node->leafgraph;
    round_visit++;
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>, CompareByFirst> candidate_set;

    float LBGRAPH;

    // add entrypoint to top_candidates &&  candidate_set
    float dist = g->fstdistfunc_(data_point, g->getDataByInternalId(entrypoint), g->dist_func_param_);
    int entrypointLabel=g->getExternalLabel(entrypoint);

#ifdef CALC_DC
    stats.distance_computations_bsl++; 
#endif
    LBGRAPH = dist;                           // distance between query point and enterpoint in the flat level
    top_candidates.emplace(dist, entrypointLabel); // max-heap (dist,label)
    candidate_set.emplace(-dist, entrypoint); // (dist,internalID)
    threadvisits[entrypoint] = round_visit; ////////////////////////////////

    while (candidate_set.size() > 0) {

        auto currnode = candidate_set.top();

        if ((-currnode.first) > LBGRAPH) {
            break;
        }
        candidate_set.pop();

        unsigned int currnodeid = currnode.second;
        int *data = (int *) g->get_linklist0(currnodeid);
        size_t neighborhood = g->getListCount((unsigned int*)data);


        _mm_prefetch((char *) (threadvisits + *(data + 1)), _MM_HINT_T0); // *(data + 1): data[1]   ||  threadvisits + *(data + 1):  threadvisits[data[1]]
        _mm_prefetch((char *) (threadvisits + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(g->data_level0_memory_ + (*(data + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

        for (size_t j = 1; j <= neighborhood; j++) {
            int neighborid = *(data + j);

            _mm_prefetch((char *) (threadvisits + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(g->data_level0_memory_ + (*(data + j + 1)) * g->size_data_per_element_ + g->offsetData_,  _MM_HINT_T0);
            if (threadvisits[neighborid] == round_visit) continue; // 检查当前邻居是否已经被访问过


            threadvisits[neighborid] = round_visit;

            char *currObj1 = (g->getDataByInternalId(neighborid));
            int currObjLable= g->getExternalLabel(neighborid);
            auto dist = g->fstdistfunc_(data_point, currObj1, g->dist_func_param_);
#ifdef CALC_DC
                stats.distance_computations_bsl++;
#endif
            if(dist < bsf)bsf=dist;


            if (top_candidates.size() < beamwidth|| LBGRAPH > dist) {
                    candidate_set.emplace(-dist, neighborid);

                    _mm_prefetch(g->data_level0_memory_ + candidate_set.top().second * g->size_data_per_element_ + g->offsetLevel0_,  _MM_HINT_T0);////////////////////////


                    // top_candidates.emplace(dist, neighborid);
                    top_candidates.emplace(dist, currObjLable);

                    if (top_candidates.size() > beamwidth)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        LBGRAPH = top_candidates.top().first;

            }
            }
        }
    }



/* 迭代训练bottom layer 边权 */
void QueryEngine::TrainWeightinflat(Node * node, unsigned int entrypoint, const void *data_point, size_t beamwidth,size_t k,
                std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> & top_candidates2,
                float & bsf,querying_stats & stats, unsigned short *threadvisits, unsigned short & round_visit, IterRefinement_epoch ep,
                int* groundtruth_id){

        std::vector<unsigned int> hop_path1;
        std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> top_candidates_internalID1;
        std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates1;

        std::vector<unsigned int> hop_path2;
        std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> top_candidates_internalID2;

        // first epoch
        if(ep==0){
            // searchflatWithHopPath(node, entrypoint, data_point, std::max(beamwidth/2, k), k , top_candidates1, top_candidates_internalID1, bsf, stats, threadvisits, round_visit, hop_path1); // 差的结果配置
            searchflatWithHopPath(node, entrypoint, data_point, std::max(beamwidth, k), k, top_candidates2, top_candidates_internalID2, bsf, stats, threadvisits, round_visit, hop_path2); // 好的结果配置
            // updateWeightByHopPath(node, data_point, hop_path1, hop_path2, top_candidates_internalID1, top_candidates_internalID2, groundtruth_id);
            updateWeightByHopPath(node, data_point, hop_path2,  top_candidates_internalID2, groundtruth_id);

        }else{
            // 退火温度
            float T =  1.5f * pow(0.95f, ep); // T0=1.5, gamma=0.95
            float thres_probabilit = 0.0f;

            // 以 entrypoint 的邻接边权作为该叶子节点本轮 μ 的估计依据，匹配目标度 M*
            if(node->leafgraph == nullptr) {
                std::cerr << "Warning: TrainWeightinflat called on leaf with null leafgraph, skipping..." << std::endl;
                return;
            }
            auto g = node->leafgraph;
            std::vector<double> local_edge_weights;
            int *data_mu = (int *)g->get_linklist0(entrypoint);
            size_t neighborhood_mu = g->getListCount((unsigned int*)data_mu);
            for (size_t j = 1; j <= neighborhood_mu; j++) {
                int neighborid = *(data_mu + j);
                Edge &edge_ref = g->edgeWeightList_[entrypoint][neighborid];
                local_edge_weights.push_back(static_cast<double>(edge_ref.weight));
            }
            double target_degree = static_cast<double>(node->leafgraph->M_); // 若无 M_ 可换为 index->index_setting->M
            if (target_degree <= 0.0) target_degree = neighborhood_mu; // 兜底
            float μ = static_cast<float>(grasp_find_mu_by_bisection(local_edge_weights, T, target_degree));

            // searchflatWithWeight_HopPath(node, entrypoint, data_point, std::max(beamwidth/2, k), k, top_candidates1, top_candidates_internalID1,
            //                             bsf, stats, threadvisits,round_visit, hop_path1, μ, T, thres_probabilit); // 差的结果配置

            searchflatWithWeight_HopPath(node, entrypoint, data_point, std::max(beamwidth, k), k ,top_candidates2, top_candidates_internalID2,
                                        bsf, stats, threadvisits,round_visit, hop_path2, μ, T, thres_probabilit);

            // updateWeightByHopPath(node, data_point, hop_path1, hop_path2, top_candidates_internalID1, top_candidates_internalID2, groundtruth_id);
            // updateWeightByHopPath(node, data_point, hop_path1, hop_path2, top_candidates_internalID1, top_candidates_internalID2);
            updateWeightByHopPath(node, data_point, hop_path2, top_candidates_internalID2, groundtruth_id);
        }
}
unsigned int getTop1Index(std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float, unsigned int>>> top_candidates) {
    // 创建队列的副本，避免破坏原始队列
    auto tmp_queue = top_candidates;  // 复制构造函数
    
    // 临时保存最大堆的元素
    std::vector<std::pair<float, unsigned int>> candidates;

    // 将副本队列中的元素转移到一个临时容器
    while (!tmp_queue.empty()) {
        candidates.push_back(tmp_queue.top());
        tmp_queue.pop();
    }

    // 返回距离最小的元素的节点下标(internal ID)
    return candidates.back().second;
}


// 原始版本
/*void QueryEngine::searchflatWithHopPath(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates_internalID,
                float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                std::vector<unsigned int> &hop_path) { // Add hop_path to capture the top-1 to entrypoint path
    auto g = node->leafgraph;
    round_visit++;
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>, CompareByFirst> candidate_set;

    float LBGRAPH;

    // Create a map to store parent-child relationships
    std::unordered_map<unsigned int, unsigned int> parent_map; // Key: current node internal ID, Value: parent node internal ID

    // Add entrypoint to top_candidates && candidate_set
    float dist = g->fstdistfunc_(data_point, g->getDataByInternalId(entrypoint), g->dist_func_param_);
    int entrypointLabel=g->getExternalLabel(entrypoint);
#ifdef CALC_DC
    stats.distance_computations_bsl++; 
#endif
    LBGRAPH = dist;                           // distance between query point and entrypoint in the flat level
    top_candidates.emplace(dist, entrypointLabel); // max-heap
    top_candidates_internalID.emplace(dist, entrypoint);
    candidate_set.emplace(-dist, entrypoint); // 
    threadvisits[entrypoint] = round_visit; ////////////////////////////////


    while (candidate_set.size() > 0) {
        auto currnode = candidate_set.top();

        if ((-currnode.first) > LBGRAPH)  break;
        candidate_set.pop();

        // center point 
        unsigned int currnodeid = currnode.second;
        int currnodeLabel= g->getExternalLabel(currnodeid);

        int *data = (int *)g->get_linklist0(currnodeid);
        size_t neighborhood = g->getListCount((unsigned int*)data);

        _mm_prefetch((char *) (threadvisits + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (threadvisits + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(g->data_level0_memory_ + (*(data + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

        // neigbors of center point 
        for (size_t j = 1; j <= neighborhood; j++) {
            int neighborid = *(data + j);

            _mm_prefetch((char *) (threadvisits + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(g->data_level0_memory_ + (*(data + j + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
            if (threadvisits[neighborid] == round_visit) continue;

            threadvisits[neighborid] = round_visit;

            char *currObj1 = (g->getDataByInternalId(neighborid));
            int currObjLable= g->getExternalLabel(neighborid);
            auto dist = g->fstdistfunc_(data_point, currObj1, g->dist_func_param_);
#ifdef CALC_DC
            stats.distance_computations_bsl++;
#endif
            if (dist < bsf) bsf = dist;

            if (top_candidates_internalID.size() < beamwidth || LBGRAPH > dist) {
                candidate_set.emplace(-dist, neighborid);

                _mm_prefetch(g->data_level0_memory_ + candidate_set.top().second * g->size_data_per_element_ +
                             g->offsetLevel0_, _MM_HINT_T0);

                top_candidates.emplace(dist, currObjLable);
                top_candidates_internalID.emplace(dist, neighborid);

                // 修复：同步裁剪两个队列，保持一致性
                if (top_candidates_internalID.size() > beamwidth) {
                    top_candidates.pop();
                    top_candidates_internalID.pop(); 
                }

                if (!top_candidates_internalID.empty())
                    LBGRAPH = top_candidates_internalID.top().first;


                parent_map[neighborid] = currnodeid; // Store the parent node

            }
        }
    }
    while (top_candidates.size() > k)top_candidates.pop();
    while (top_candidates_internalID.size() > k)top_candidates_internalID.pop();

    // Now backtrack from top-1 node to entrypoint
    unsigned int current_node = getTop1Index(top_candidates_internalID);    
    std::vector<unsigned int> final_path;
    
    // Backtrack the path from the best node to entrypoint using parent_map
    while (current_node != entrypoint) {
        final_path.push_back(current_node);
        if (parent_map.find(current_node) == parent_map.end()) {
            throw std::runtime_error("Error: parent_map does not contain node " + std::to_string(current_node));
        }
        current_node = parent_map.at(current_node);
    }
    
    // Add entrypoint to complete the path
    final_path.push_back(entrypoint);

    // Reverse the path to get it from entrypoint to top-1 node
    std::reverse(final_path.begin(), final_path.end());
    hop_path = final_path;
}
*/

// 修改版本
void QueryEngine::searchflatWithHopPath(
    Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates_internalID,
    float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
    std::vector<unsigned int> &hop_path) 
{
    auto g = node->leafgraph;
    round_visit++;

    // min-heap via negative dist (不变)
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>, CompareByFirst> candidate_set;

    float LBGRAPH;

    // 记录父子关系以回溯 hop path
    std::unordered_map<unsigned int, unsigned int> parent_map;
    // tbb::concurrent_unordered_map<unsigned int, unsigned int> parent_map;

    // ---- 初始化：仅维护 internalID 堆 ----
    float dist = g->fstdistfunc_(data_point, g->getDataByInternalId(entrypoint), g->dist_func_param_);
    int entrypointLabel = g->getExternalLabel(entrypoint);
#ifdef CALC_DC
    stats.distance_computations_bsl++;
#endif
    LBGRAPH = dist;

    // 初始化：两堆都放入 entrypoint，但后续以 internalID 堆为唯一控制来源
    top_candidates_internalID.emplace(dist, entrypoint);                   // max-heap by dist
    top_candidates.emplace(dist, static_cast<unsigned int>(entrypointLabel));
    candidate_set.emplace(-dist, entrypoint);                              // 小根 via 负号
    threadvisits[entrypoint] = round_visit;

    // 主循环
    while (candidate_set.size() > 0) {
        auto currnode = candidate_set.top();

        if ((-currnode.first) > LBGRAPH) break;
        candidate_set.pop();

        unsigned int currnodeid = currnode.second;
        int *data = (int *)g->get_linklist0(currnodeid);
        size_t neighborhood = g->getListCount((unsigned int*)data);

        _mm_prefetch((char *)(threadvisits + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(threadvisits + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(g->data_level0_memory_ + (*(data + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *)(data + 2), _MM_HINT_T0);

        for (size_t j = 1; j <= neighborhood; j++) {
            int neighborid = *(data + j);

            _mm_prefetch((char *)(threadvisits + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(g->data_level0_memory_ + (*(data + j + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);

            if (threadvisits[neighborid] == round_visit) continue;
            threadvisits[neighborid] = round_visit;

            char *currObj1 = g->getDataByInternalId(neighborid);
            auto dist = g->fstdistfunc_(data_point, currObj1, g->dist_func_param_);
#ifdef CALC_DC
            stats.distance_computations_bsl++;
#endif
            if (dist < bsf) bsf = dist;

            // 控制逻辑仅依赖 internalID 堆；但两堆都同步插入/裁剪
            if (top_candidates_internalID.size() < beamwidth || LBGRAPH > dist) {
                candidate_set.emplace(-dist, neighborid);
                _mm_prefetch(g->data_level0_memory_ + candidate_set.top().second * g->size_data_per_element_ + g->offsetLevel0_, _MM_HINT_T0);

                top_candidates_internalID.emplace(dist, neighborid);
                top_candidates.emplace(dist, static_cast<unsigned int>(g->getExternalLabel(neighborid)));

                if (top_candidates_internalID.size() > beamwidth) top_candidates_internalID.pop();
                if (top_candidates.size() > beamwidth) top_candidates.pop();

                if (!top_candidates_internalID.empty()) LBGRAPH = top_candidates_internalID.top().first;

                parent_map[neighborid] = currnodeid; // 记录父子关系
            }
        }
    }

    // 截到 top-k（internalID 堆）
    while (top_candidates_internalID.size() > k) top_candidates_internalID.pop();
    while (top_candidates.size() > k) top_candidates.pop();

    // --- 回溯 hop path ---
    unsigned int current_node = getTop1Index(top_candidates_internalID);
    std::vector<unsigned int> final_path;
    while (current_node != entrypoint) {
        final_path.push_back(current_node);
        auto it = parent_map.find(current_node);
        if (it == parent_map.end()) {
            throw std::runtime_error("Error: parent_map does not contain node " + std::to_string(current_node));
        }
        current_node = it->second;
    }
    final_path.push_back(entrypoint);
    std::reverse(final_path.begin(), final_path.end());
    hop_path = std::vector<unsigned int>(final_path);

    // 两堆已同步裁剪，无需重建
}

/* // 原始版本
void QueryEngine::updateWeightByHopPath(Node *node, const void *data_point, 
                            std::vector<unsigned int> hop_path1, std::vector<unsigned int> hop_path2,
                            std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates1,
                            std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>> top_candidates2){
    
    unsigned int top1 = getTop1Index(top_candidates1);
    unsigned int top2 = getTop1Index(top_candidates2);
    if(top1 == top2) return;

    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: updateWeightByHopPath called on leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    
    auto g = node->leafgraph;
    float dist1 = g->fstdistfunc_(data_point, g->getDataByInternalId(top1), g->dist_func_param_);
    float dist2 = g->fstdistfunc_(data_point, g->getDataByInternalId(top2), g->dist_func_param_);
    
    // 添加安全检查，避免除零
    if(std::fabs(dist2) < 1e-12f) return;
    
    // 计算权重更新，添加范围限制
    float ratio = dist1 / dist2;
    float new_weight = (ratio - 1.0f) * 0.1f ; // learning_rate=1.0f
    
    if(ratio<1){  // top1更好，增强hop_path1
        for (size_t i = 0; i < hop_path1.size() - 1; ++i) {
            unsigned int current_node = hop_path1[i];
            unsigned int next_node = hop_path1[i + 1];
            
            if(g->edgeWeightList_ == nullptr) throw std::runtime_error("g->edgeWeightList_==nullptr");
            
            Edge &current_edge = g->edgeWeightList_[current_node][next_node];
            
            current_edge.weight+= -new_weight;
        }

    }else{
        for (size_t i = 0; i < hop_path2.size() - 1; ++i) {
            unsigned int current_node = hop_path2[i];
            unsigned int next_node = hop_path2[i + 1];
            
            if(g->edgeWeightList_ == nullptr) throw std::runtime_error("g->edgeWeightList_==nullptr");
            
            Edge &current_edge = g->edgeWeightList_[current_node][next_node];

            current_edge.weight += new_weight;
        }
    }

}
 */

// 改进版本2：基于真实最近邻的监督权重更新
void QueryEngine::updateWeightByHopPath( Node *node, const void *data_point, 
                            std::vector<unsigned int> hop_path,
                            std::priority_queue<std::pair<float,unsigned int>, std::vector<std::pair<float,unsigned int>>>  top_candidates,
                            int* groundtruth_id) {
    
    // 安全检查
    if(node->leafgraph == nullptr) {
        throw std::runtime_error("Warning: updateWeightByHopPath called on leaf with null leafgraph, skipping...");
    }
    auto g = node->leafgraph;
    
    unsigned int top1 = getTop1Index(top_candidates); 
    
    // 2. 计算距离用于质量评估
    float dist1 = g->fstdistfunc_(data_point, g->getDataByInternalId(top1), g->dist_func_param_);

    const int dim = this->index->index_setting->timeseries_size;
    if (groundtruth_id == nullptr) return; // nothing to compare
    int gt_id = groundtruth_id[0];
    if (gt_id < 0) return;

    static thread_local std::vector<ts_type> gt_buffer;
    if ((int)gt_buffer.size() != dim) gt_buffer.resize(dim);

    FILE *dataset_file = fopen(this->dataset, "rb");
    if (dataset_file == nullptr) {
        fprintf(stderr, "Dataset file %s not found! (skip weight update)\n", this->dataset);
        return; // graceful degradation instead of exit
    }
    // Seek & read groundtruth vector
    if (fseek(dataset_file, (long)gt_id * dim * sizeof(ts_type), SEEK_SET) != 0) {
        fclose(dataset_file);
        return; // seek error, skip update
    }
    size_t read_cnt = fread(gt_buffer.data(), sizeof(ts_type), dim, dataset_file);
    fclose(dataset_file);
    if (read_cnt != (size_t)dim) return; // failed read, skip

    float top1_distance = 0.0f;
    const ts_type* q = static_cast<const ts_type*>(data_point);
    for (int i = 0; i < dim; i++) {
        float diff = (float)q[i] - (float)gt_buffer[i];
        top1_distance += diff * diff;
    }
    if (top1_distance <= 1e-12f) return; // avoid divide-by-zero below

    // 计算dist1与top1_distance的差距，差距越小，权重更新越大
    float diff_ratio = std::fabs(dist1 - top1_distance) / top1_distance;
    float new_weight =  1.0f / (1.0f + diff_ratio); // learning_rate=1.0f
    for (size_t i = 0; i < hop_path.size() - 1; ++i) {
        unsigned int current_node = hop_path[i];
        unsigned int next_node = hop_path[i + 1];
        
        if(g->edgeWeightList_ == nullptr) throw std::runtime_error("g->edgeWeightList_==nullptr");
        
        Edge &current_edge = g->edgeWeightList_[current_node][next_node];
        
        current_edge.weight+= new_weight;
    }
}


// 辅助函数：更新路径权重
void QueryEngine::updatePathWeights(hnswlib::HierarchicalNSW<float>* g, 
                                   const std::vector<unsigned int>& hop_path, 
                                   float weight_update, 
                                   float momentum_decay, 
                                   float weight_bound) {
    for (size_t i = 0; i < hop_path.size() - 1; ++i) {
        unsigned int current_node = hop_path[i];
        unsigned int next_node = hop_path[i + 1];
        
        if(g->edgeWeightList_==nullptr) {
            throw std::runtime_error("g->edgeWeightList_==nullptr");
        }
        
        Edge &current_edge = g->edgeWeightList_[current_node][next_node];
        
        // 应用动量衰减
        current_edge.weight *= momentum_decay;
        
        // 添加权重更新
        current_edge.weight += weight_update;
        
        // 边界限制
        current_edge.weight = std::max(-weight_bound, std::min(weight_bound, current_edge.weight));
    }
}

void  QueryEngine::searchflatWithWeight(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                          float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                          float thres_probability, float μ, float T) {
    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: searchflatWithWeight called on leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    auto g = node->leafgraph;
    round_visit++;
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>, CompareByFirst> candidate_set;

    float LBGRAPH;

    // Add entrypoint to top_candidates && candidate_set
    float dist = g->fstdistfunc_(data_point, g->getDataByInternalId(entrypoint), g->dist_func_param_);
    int entrypointLabel=g->getExternalLabel(entrypoint);
    
    #ifdef CALC_DC
    stats.distance_computations_bsl++; 
#endif
    LBGRAPH = dist;                           // distance between query point and entrypoint in the flat level
    top_candidates.emplace(dist, entrypointLabel); // max-heap
    candidate_set.emplace(-dist, entrypoint); // 
    threadvisits[entrypoint] = round_visit;  // (QueryEngine.cpp:1060)


    while (candidate_set.size() > 0) {
        auto currnode = candidate_set.top();

        if ((-currnode.first) > LBGRAPH) {
            break;
        }
        candidate_set.pop();

        unsigned int currnodeid = currnode.second;
        int *data = (int *)g->get_linklist0(currnodeid);
        size_t neighborhood = g->getListCount((unsigned int*)data);

        _mm_prefetch((char *) (threadvisits + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (threadvisits + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(g->data_level0_memory_ + (*(data + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

        for (size_t j = 1; j <= neighborhood; j++) {
            int neighborid = *(data + j);

            // Pre-fetch if necessary
            _mm_prefetch((char *) (threadvisits + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(g->data_level0_memory_ + (*(data + j + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);

            // Check if this neighbor has already been visited
            if (threadvisits[neighborid] == round_visit) continue;

            // Calculate edge weight (probability) for the current edge  currnodeid->neighborid
            Edge *current_edge = &g->edgeWeightList_[currnodeid][neighborid];
            float edge_prob = 1.0f / (1 + exp(-((current_edge->weight + μ) / T)));
            if (edge_prob <= thres_probability) {
                // 对权重为0的边按概率 ρ 放行
                if (std::fabs(current_edge->weight) <= 1e-12f) {
                    thread_local std::mt19937 rng(
                        std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ static_cast<uint32_t>(std::time(nullptr))
                    );
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    if (dist(rng) > this->zero_edge_pass_ratio) {
                        continue;
                    }
                }
            }

            threadvisits[neighborid] = round_visit; // marked visited

            char *currObj1 = (g->getDataByInternalId(neighborid));
            int currObjLable= g->getExternalLabel(neighborid);
            auto dist = g->fstdistfunc_(data_point, currObj1, g->dist_func_param_);
#ifdef CALC_DC
            stats.distance_computations_bsl++;
#endif
            if (dist < bsf) bsf = dist; // update global bsf

            if (top_candidates.size() < beamwidth || LBGRAPH > dist) {
                candidate_set.emplace(-dist, neighborid);

                _mm_prefetch(g->data_level0_memory_ + candidate_set.top().second * g->size_data_per_element_ + g->offsetLevel0_, _MM_HINT_T0);

                top_candidates.emplace(dist, currObjLable);

                if (top_candidates.size() > beamwidth)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    LBGRAPH = top_candidates.top().first;
            }
        }
    }
    while (top_candidates.size() > k)top_candidates.pop();
}

void QueryEngine::searchflatWithWeight_HopPath(Node *node, unsigned int entrypoint, const void *data_point, size_t beamwidth, size_t k,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                          std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates_internalID,
                          float &bsf, querying_stats &stats, unsigned short *threadvisits, unsigned short &round_visit,
                          std::vector<unsigned int> &hop_path, float μ, float T,float thres_probability) {
    auto g = node->leafgraph;
    round_visit++;
    std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>, CompareByFirst> candidate_set;

    float LBGRAPH;

    // Create a map to store parent-child relationships // Key: current node, Value: parent node
    std::unordered_map<unsigned int, unsigned int> parent_map; 

    // Add entrypoint to top_candidates && candidate_set
    float dist = g->fstdistfunc_(data_point, g->getDataByInternalId(entrypoint), g->dist_func_param_);
    int entrypointLabel=g->getExternalLabel(entrypoint);
    
    #ifdef CALC_DC
    stats.distance_computations_bsl++; 
#endif
    LBGRAPH = dist;                           // distance between query point and entrypoint in the flat level
    top_candidates.emplace(dist, entrypointLabel); // max-heap
    top_candidates_internalID.emplace(dist, entrypoint);
    candidate_set.emplace(-dist, entrypoint); // 
    // threadvisits[round_visit] = round_visit;  // (QueryEngine.cpp:1060)
    threadvisits[entrypoint] = round_visit;  // (QueryEngine.cpp:1060)


    while (candidate_set.size() > 0) {
        auto currnode = candidate_set.top();

        if ((-currnode.first) > LBGRAPH) {
            break;
        }
        candidate_set.pop();

        unsigned int currnodeid = currnode.second;
        int *data = (int *)g->get_linklist0(currnodeid);
        size_t neighborhood = g->getListCount((unsigned int*)data);

        _mm_prefetch((char *) (threadvisits + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (threadvisits + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(g->data_level0_memory_ + (*(data + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);
        _mm_prefetch((char *) (data + 2), _MM_HINT_T0);

        for (size_t j = 1; j <= neighborhood; j++) {
            int neighborid = *(data + j);

            // Pre-fetch if necessary
            _mm_prefetch((char *) (threadvisits + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(g->data_level0_memory_ + (*(data + j + 1)) * g->size_data_per_element_ + g->offsetData_, _MM_HINT_T0);

            // Check if this neighbor has already been visited
            if (threadvisits[neighborid] == round_visit) continue;

            // Calculate edge weight (probability) for the current edge
            Edge *current_edge = &g->edgeWeightList_[currnodeid][neighborid];
            float edge_prob = 1.0f / (1 + exp(-((current_edge->weight + μ) / T)));
            if (edge_prob <= thres_probability) {
                if (std::fabs(current_edge->weight) <= 1e-12f) {
                    thread_local std::mt19937 rng(
                        std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ static_cast<uint32_t>(std::time(nullptr))
                    );
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    if (dist(rng) > this->zero_edge_pass_ratio) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            threadvisits[neighborid] = round_visit; // marked visited

            char *currObj1 = (g->getDataByInternalId(neighborid));
            int currObjLable= g->getExternalLabel(neighborid);
            auto dist = g->fstdistfunc_(data_point, currObj1, g->dist_func_param_);
#ifdef CALC_DC
            stats.distance_computations_bsl++;
#endif
            if (dist < bsf) bsf = dist; // update global bsf

            if (top_candidates.size() < beamwidth || LBGRAPH > dist) {
                candidate_set.emplace(-dist, neighborid);

                _mm_prefetch(g->data_level0_memory_ + candidate_set.top().second * g->size_data_per_element_ +
                             g->offsetLevel0_, _MM_HINT_T0);

                top_candidates.emplace(dist, currObjLable);
                top_candidates_internalID.emplace(dist, neighborid);


                if (top_candidates.size() > beamwidth)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    LBGRAPH = top_candidates.top().first;

                parent_map[neighborid] = currnodeid; // Store the parent node

            }
        }
    }
    while (top_candidates.size() > k)top_candidates.pop();
    while (top_candidates_internalID.size() > k)top_candidates_internalID.pop();  

    // Now backtrack from top-1 node to entrypoint
    unsigned int current_node = getTop1Index(top_candidates_internalID);    
    std::vector<unsigned int> final_path;

    
    // Backtrack the path from the best node to entrypoint using parent_map
    while (current_node != entrypoint) {
        final_path.push_back(current_node);
        if (parent_map.find(current_node) == parent_map.end()) {
            throw std::runtime_error("Error: parent_map does not contain node " + std::to_string(current_node));
        }
        current_node = parent_map.at(current_node);
    }
    
    // Add entrypoint to complete the path
    final_path.push_back(entrypoint);

    // Reverse the path to get it from entrypoint to top-1 node
    std::reverse(final_path.begin(), final_path.end());
    hop_path = final_path;
}



void QueryEngine::searchGraphLeaf(Node * node,const void *query_data, size_t k,
                     std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                     float &bsf, querying_stats &stats, unsigned short *flags, unsigned short & flag, bool searchWithWeight, float thres_probability, float μ, float T)  {
    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: trying to search in leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    auto g = node->leafgraph;
    unsigned int currObj = g->enterpoint_node_;
    float curdist = g->fstdistfunc_(query_data, g->getDataByInternalId(currObj), g->dist_func_param_);

    // search the top level 
    for (int level = g->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;
            data = (unsigned int *) g->get_linklist(currObj, level);
            int size = g->getListCount(data);
            unsigned int *datal = (unsigned int *) (data + 1);
            for (int i = 0; i < size; i++) {
                unsigned int cand = datal[i];
                if (cand < 0 || cand > g->max_elements_)
                    throw std::runtime_error("cand error");
#ifdef CALC_DC
                stats.distance_computations_hrl++; // 计数器，统计距离计算的次数
#endif
                float d = g->fstdistfunc_(query_data, g->getDataByInternalId(cand), g->dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    // search the flat level 
    if(searchWithWeight){
        searchflatWithWeight(node, currObj, query_data, std::max(g->ef_, k), k , top_candidates, bsf, stats, flags, flag, thres_probability, μ, T);
    }else{
        searchflat(node, currObj, query_data, std::max(g->ef_, k), k , top_candidates, bsf, stats, flags, flag);
    }
    while (top_candidates.size() > k)top_candidates.pop();
};



/* 
* query_ts: query
* k: top-k vector candidates
* nprobes: candidate leaf number
*/
void QueryEngine::TrainWeightinGraphLeaf(Node * node,const void *query_data, size_t k,
                     std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                     float &bsf, querying_stats &stats, unsigned short *flags, unsigned short & flag, IterRefinement_epoch ep_index,
                     int* groundtruth_id)  {

    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: trying to train weight in leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    auto g = node->leafgraph;
    unsigned int currObj = g->enterpoint_node_;
    float curdist = g->fstdistfunc_(query_data, g->getDataByInternalId(currObj), g->dist_func_param_);

    // search the top level 
    for (int level = g->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;
            data = (unsigned int *) g->get_linklist(currObj, level);
            if(data==nullptr){
                throw std::runtime_error("data==nullptr");
                exit(-1);
            }
            int size = g->getListCount(data);
            unsigned int *datal = (unsigned int *) (data + 1);
            for (int i = 0; i < size; i++) {
                unsigned int cand = datal[i];
                if (cand < 0 || cand > g->max_elements_)
                    throw std::runtime_error("cand error");
#ifdef CALC_DC
                stats.distance_computations_hrl++; // 计数器，统计距离计算的次数
#endif
                float d = g->fstdistfunc_(query_data, g->getDataByInternalId(cand), g->dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    // search the flat level 
    TrainWeightinflat(node, currObj, query_data, std::max(g->ef_, k), k , top_candidates, bsf, stats, flags, flag, ep_index, groundtruth_id);  // efSearch > k 才有意义

};

void QueryEngine::searchWithWeightinGraphLeaf(Node * node,const void *query_data, size_t k,
                     std::priority_queue<std::pair<float, unsigned int>, std::vector<std::pair<float, unsigned int>>> &top_candidates,
                     float &bsf, querying_stats &stats, unsigned short *flags, unsigned short & flag,
                     bool search_withWeight, float thres_probability, float μ, float T, int* groundtruth_id)  {

    if(node->leafgraph == nullptr) {
        std::cerr << "Warning: trying to train weight in leaf with null leafgraph, skipping..." << std::endl;
        return;
    }
    auto g = node->leafgraph;
    unsigned int currObj = g->enterpoint_node_;
    float curdist = g->fstdistfunc_(query_data, g->getDataByInternalId(currObj), g->dist_func_param_);

    // search the top level 
    for (int level = g->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data;
            data = (unsigned int *) g->get_linklist(currObj, level);
            if(data==nullptr){
                throw std::runtime_error("data==nullptr");
                exit(-1);
            }
            int size = g->getListCount(data);
            unsigned int *datal = (unsigned int *) (data + 1);
            for (int i = 0; i < size; i++) {
                unsigned int cand = datal[i];
                if (cand < 0 || cand > g->max_elements_)
                    throw std::runtime_error("cand error");
#ifdef CALC_DC
                stats.distance_computations_hrl++; // 计数器，统计距离计算的次数
#endif
                float d = g->fstdistfunc_(query_data, g->getDataByInternalId(cand), g->dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    if(search_withWeight)
        searchflatWithWeight(node, currObj, query_data, std::max(g->ef_, k), k , top_candidates, bsf, stats, flags, flag, 
                        thres_probability, μ, T);  // efSearch > k 才有意义
    else
        searchflat(node, currObj, query_data, std::max(g->ef_, k), k , top_candidates, bsf, stats, flags, flag);
};

void QueryEngine::TrainWeightinNpLeafParallel(ts_type *query_ts, int *groundtruth_id,unsigned int k, unsigned int nprobes, unsigned int learn_index, 
                                            IterRefinement_epoch ep_index,std::set<Node*> candidate_leaf) {

    stats.reset();

    Time start = now();

    ts_type kth_bsf = FLT_MAX;  // global  kth_bsf

    //  分配前先释放之前的内存（如果存在）
    // 在分配前添加
    cout<<"************************************************"<<endl;
    printf("ep_index=%d, learn_index=%d, releasing previous memory: %s\n", ep_index, learn_index, (this->learn_results[learn_index] != nullptr) ? "yes" : "no");
    if (this->learn_results[learn_index] != nullptr) {
        free(this->learn_results[learn_index]);
        this->learn_results[learn_index] = nullptr;
    }
    
    this->learn_results[learn_index] = (int*)calloc(k, sizeof(int));
    if (this->learn_results[learn_index] == nullptr) {
        fprintf(stderr, "Memory allocation failed for learn_results[%d]\n", learn_index);
        return;
    }

    //将集合candidate_leaf_node中的元素插入到candidates
    Node ** candidates =  static_cast<Node **>(calloc(Node::num_leaf_node+1, sizeof(Node *)));
    if (candidates == nullptr) {
        fprintf(stderr, "Memory allocation failed for candidates\n");
        return;
    }
    
    int i = 0;
    for (Node* node_ptr : candidate_leaf) {
        if (i < Node::num_leaf_node) { // 修复边界条件：使用 < 而不是 <=
            candidates[i] = node_ptr;  // 只保存指针
            i++;
        }
    }
    unsigned int candidates_count = candidate_leaf.size();
    stats.num_candidates = candidate_leaf.size();


    if (parallel and nprobes > 1) { 
        for (int i = 1; i < nworker; i++) {
            qwdata[i].id = i;
            qwdata[i].kth_bsf = &kth_bsf;
            qwdata[i].stats->reset();
            qwdata[i].bsf = FLT_MAX;
        }
        qwdata[0].kth_bsf = &kth_bsf;   // global kth_bsf
        qwdata[0].bsf = FLT_MAX;
        qwdata[0].id=0;
        qwdata[0].flags = flags;
        qwdata[0].curr_flag = curr_flag;

        // copypq(qwdata, top_candidates); // top_candidates复制到每个 pData[i].top_candidates


        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER; // 声明并初始化一个读写锁

        // 声明并行处理中使用的变量
        Node* node;
        query_worker_data *worker;

    // 3.2 开始并行搜索
        {
            #pragma omp parallel num_threads(nworker) private(node, worker) shared(qwdata, candidates_count, candidates,  query_ts, k)
            {
                worker = qwdata+omp_get_thread_num();

                // 3.2.1 每个线程处理一个循环迭代
                #pragma omp for schedule(static, 1) 
                // for (int i = 0; i < std::min(candidates_count,nprobes); i++) { 
                for (int i = 0; i < candidates_count; i++) {    
                    node = candidates[i];
                    TrainWeightinGraphLeaf(node, query_ts,  k, *(worker->top_candidates), worker->bsf, 
                                     *(worker->stats), worker->flags, worker->curr_flag, ep_index, groundtruth_id);
                }
            }
        }

        for(int i=0; i<nworker; i++){
            /* 对多线程结果进行合并 */
            while(qwdata[i].top_candidates->size()>0){
                top_candidates.emplace(qwdata[i].top_candidates->top());
                qwdata[i].top_candidates->pop();
            }
        }

    } 

    while (top_candidates.size() > k) top_candidates.pop();
    float recall2 = calculateRecall(top_candidates, groundtruth_id, k, groundtruth_top_k);
    cout<<"recall2:"<<recall2<<endl;

    /* 统计topkID */
    int m=0;
    while (top_candidates.size() > 0) {
            this->learn_results[learn_index][m]=top_candidates.top().second;
            m++;
            top_candidates.pop();
    }

    double time = getElapsedTime(start);
    
    // 4.5 清理内存
    if (candidates != nullptr) {
        free(candidates);
        candidates = nullptr;
    }
}


void QueryEngine::searchWithWeightinNpLeafParallel(ts_type *query_ts, int *groundtruth_id,unsigned int k, unsigned int nprobes, unsigned int query_index,
                                                 bool search_withWeight, float thres_probability,float μ, float T, std::set<Node*> candidate_leaf) {

    stats.reset();
    this->results[query_index]=(int*)calloc(k, sizeof(int));
    if (this->results[query_index] == nullptr) {
        fprintf(stderr, "Memory allocation failed for this->results[%d]\n", query_index);
        exit(-1);
    }

    //将集合candidate_leaf_node中的元素插入到candidates
    Node ** candidates =  static_cast<Node **>(calloc(Node::num_leaf_node+1, sizeof(Node *)));
    if (candidates == nullptr) {
        fprintf(stderr, "Memory allocation failed for candidates\n");
        return;
    }
    
    int i = 0;
    for (Node* node_ptr : candidate_leaf) {
        if (i < Node::num_leaf_node) { 
            candidates[i] = node_ptr; 
            i++;
        }
    }
    unsigned int candidates_count = candidate_leaf.size();
    stats.num_candidates = candidate_leaf.size();


    ts_type kth_bsf = FLT_MAX;  // global  kth_bsf
    Time start = now();
    if (parallel and nprobes > 1) { 

        for (int i = 1; i < nworker; i++) {
            qwdata[i].id = i;
            qwdata[i].kth_bsf = &kth_bsf;
            qwdata[i].stats->reset();
            qwdata[i].bsf = FLT_MAX;
        }
        qwdata[0].kth_bsf = &kth_bsf;   // global kth_bsf
        qwdata[0].bsf = FLT_MAX;
        qwdata[0].id=0;
        qwdata[0].flags = flags;
        qwdata[0].curr_flag = curr_flag;

        // copypq(qwdata, top_candidates); // top_candidates复制到每个 pData[i].top_candidates

        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER; // 声明并初始化一个读写锁

        // 声明并行处理中使用的变量
        Node* node;
        query_worker_data *worker;

    // 3.2 开始并行搜索
        {
            #pragma omp parallel num_threads(nworker) private(node, worker) shared(qwdata, candidates_count, candidates,  query_ts, k)
            {
                worker = qwdata+omp_get_thread_num();

                // 3.2.1 每个线程处理一个循环迭代
                #pragma omp for schedule(static, 1) 
                // for (int i = 0; i < std::min(candidates_count,nprobes); i++) { 
                for (int i = 0; i < candidates_count; i++) {    
                    node = candidates[i];
                    searchWithWeightinGraphLeaf(node, query_ts,  k, *(worker->top_candidates), worker->bsf, 
                                     *(worker->stats), worker->flags, worker->curr_flag,
                                       search_withWeight, thres_probability, μ, T, groundtruth_id);
                }
            }
        }

        for(int i=0; i<nworker; i++){
            /* 对多线程结果进行合并 */
            while(qwdata[i].top_candidates->size()>0){
                top_candidates.emplace(qwdata[i].top_candidates->top());
                qwdata[i].top_candidates->pop();
            }
        }

    } 
    while (top_candidates.size() > k) top_candidates.pop();
    cout<<"************************************************"<<endl;
    cout<<"query_index:"<<query_index<<endl;
    float recall2 = calculateRecall(top_candidates, groundtruth_id, k, groundtruth_top_k);
    cout<<"recall2:"<<recall2<<endl;

    int m=0;
    while (top_candidates.size() > 0) {
            this->results[query_index][m]=top_candidates.top().second;
            m++;
            top_candidates.pop();
    }

    double time = getElapsedTime(start);
    
    // 4.5 清理内存
    if (candidates != nullptr) {
        free(candidates);
        candidates = nullptr;
    }
}

inline void QueryEngine::copypq(query_worker_data *pData, priority_queue<pair<float, unsigned int>> queue) {
    while(!queue.empty()){
        auto p = queue.top();
        p.second  = UINT32_MAX;  
        for(int i =1; i<nworker; i++){//0 is reserved to top_candidates itself, AND worker 0 has a reference to PQ
            pData[i].top_candidates->emplace(p);
        }
        queue.pop();
    }
}
