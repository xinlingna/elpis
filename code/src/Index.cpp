#include <unistd.h> 
#include "Index.h"
#include "file_utils.h"
// No need for forward declaration if file_utils.h defines deleteDirectory


/**
 <center><h3>Init Index => Setting and BufferManager and root node</h3>
 */
Index * Index::initIndex(char *path, unsigned int size, double size1, unsigned int segments, unsigned int size2,
                         int construction, int m) {

    return new Index(path, size, size1, segments, size2, construction, m);

}

Index::Index(char *index_path, unsigned int timeseries_size, double buffered_memory_size,
             unsigned int init_segments, unsigned int max_leaf_size, int efConstruction, int M){
    this->time_stats = new timelapse ;
    this->time_stats->index_building_time = 0;
    this->time_stats->querying_time = 0;
    this->in_memory = 0;
    if(chdir(index_path) == 0){
        cout<<"The index folder is already existing"<<index_path<< endl;
        deleteDirectory(index_path);
        // throw std::runtime_error("The index folder is already existing, Please make sure to give an non existing path to generate index within!");
    }


    mkdir(index_path , 07777);

    char * index_path_hercules = static_cast<char *>(malloc(sizeof(char) * (strlen(index_path) + 6)));
    index_path_hercules = strcpy( index_path_hercules , index_path);
    index_path_hercules  = strcat( index_path_hercules , "tree/");
    mkdir(index_path_hercules, 0777);


    char * index_path_hnsw = static_cast<char *>(malloc(sizeof(char) * (strlen(index_path) + 7)));
    index_path_hnsw = strcpy( index_path_hnsw , index_path);
    index_path_hnsw = strcat( index_path_hnsw , "graph/");
    mkdir(index_path_hnsw, 0777);

    char * index_path_txt = static_cast<char *>(malloc(sizeof(char) * (strlen(index_path) + 5)));
    index_path_txt = strcpy( index_path_txt , index_path);
    index_path_txt = strcat( index_path_txt , "txt/");
    mkdir(index_path_txt, 0777);


    this -> index_setting =new Setting(index_path_hercules, index_path_hnsw, index_path_txt, index_path, timeseries_size, init_segments, max_leaf_size, buffered_memory_size, efConstruction, M);
    this->buffer_manager = new BufferManager(this->index_setting);
    this->first_node = Node::rootNodeInit(this->index_setting);
}


/*
 * dataset: path of dataset
 * dataset_size: size of dataset
 */
void Index::buildIndexFromBinaryData(char* dataset, file_position_type dataset_size) {

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();


    // malloc memory for vwi
    VectorWithIndex* vwi = static_cast<VectorWithIndex*>(malloc_index(sizeof(VectorWithIndex)));
    vwi->ts = static_cast<ts_type *>(malloc_index(sizeof(ts_type) * this->index_setting->timeseries_size));
    if (vwi->ts == nullptr) {
        cerr << "Error in index.cpp: Could not allocate memory for ts in VectorWithIndex." << endl;
        exit(-1);
    }
    vwi->ts_index =static_cast<file_position_type *>(malloc_index(sizeof(file_position_type)));
    if (vwi->ts_index == nullptr) {
        cerr << "Error in index.cpp: Could not allocate memory for ts_index in VectorWithIndex." << endl;
        exit(-1);
    }

    // open file dataset
    FILE *ifile;
    ifile = fopen(dataset , "rb");
    if (ifile == nullptr) {
        fprintf(stderr, "Error in index.c: File %s not found!\n", dataset);
        exit(-1);
    }

    // compute total_records
    fseek(ifile, 0L, SEEK_END);
    auto sz = (file_position_type) ftell(ifile);
    file_position_type total_records = sz /(this->index_setting->timeseries_size * sizeof(ts_type));
    fseek(ifile, 0L, SEEK_SET);
    if (total_records < dataset_size) {
        fprintf(stderr, "File %s has only %llu records!\n", dataset, total_records);
        exit(-1);
    }

    // load one timeseries and build hercules index
    file_position_type ts_loaded = 0;
    while (ts_loaded < dataset_size) {

        fread(vwi->ts, sizeof(ts_type), this->index_setting->timeseries_size, ifile);
        *(vwi->ts_index) = ts_loaded;

        if (!this->insertTS(vwi)) {
            cerr << "Error in index.c:  Could not add the time series to the index."<<endl;
            exit(-1);
        }
        ts_loaded++;
    }


    // build hnsw for every node leaf
    hercules_file_map *lastP = this->buffer_manager->file_map_tail;
    Node ** nodes = static_cast<Node **>(malloc_index(Node::num_leaf_node * sizeof(Node *)));
    int i =0;
    while(lastP != nullptr){
            nodes[i++] = lastP->file_buffer->node;
            lastP = lastP->prev;
        }

#pragma omp parallel default(none) shared(i,nodes)
    {
#pragma omp for
        for(int j =0; j<i;j++)
            nodes[j]->leafToGraph(this);
    }

    // 释放内存 mode !=2 取消注释
    // for(int j =0; j<i;j++)
    //     nodes[j]->deleteFileBuffer(this);
    // free(this->buffer_manager->mem_array); 

    if (vwi != nullptr) {
        if (vwi->ts != nullptr) {
            free(vwi->ts);
            vwi->ts = nullptr;
        }

        if(vwi->ts_index != nullptr) {
            free(vwi->ts_index);
            vwi->ts_index = nullptr;
        }

        free(vwi);
        vwi = nullptr;
    }
    if (fclose(ifile)) {
        fprintf(stderr, "Error in index.cpp: Could not close the filename %s", dataset);
        exit(-1);
    }


    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    this->time_stats->index_building_time = elapsed.count();

}

/**
 <ol>
 <li>IF node is not leaf, root TS using policies and node & new ts Sketches until to reach the adequate leaf</li>
 <li>IF node reaches is leaf, add ts to the node
    <ul><li>check if node has buffer file if not init it and map it</li>
        <li>check if memarray is nearly full, if so flush data to disk</li>
        <li>copy @  where the new ts will be registrated at memarray in buffered list, and copy new ts in mem array</li>
        </ul>
 <li>IF leaf node has exceed number of permitted ts to save, we split the into two leaf</li></ol>

   \main_idea_of_splitting maximize difference between QOS of parent and QOS of new children,
   hmmm basically we try to minimize QOS (len*(difference_between_minandmax_mean² + max_std²)) of leaves,
   so we select the splitting strategy giving the max diff between qos of parent and the avg qos of
   its two new children
<br> <b>range=Qos</b>

        */
bool Index::insertTS(VectorWithIndex* vwi )  {

    ts_type * ts_new = vwi->ts;                   // timeseries 
    file_position_type* ts_index = vwi->ts_index; // ID of timeseries
    if (ts_new == nullptr) {
        cerr << "error in Index.cpp: ts_new is null" << endl;
        return FAILURE;
    }   

    Node * node = this->first_node;              // root

    while(!node->is_leaf){
        node->updateStatistics(ts_new); 

        if (node->node_split_policy_route_to_left(ts_new))
            node = node->left_child;
        else
            node = node->right_child;
    }


    if(node->is_leaf){

        node->updateStatistics(ts_new);


        // 将ts_new 加入到node->file_buffer 并且更新index->buffer_manager(如果有必要的话)
        node->addTS(this,vwi);


        if (node->node_size >= this->index_setting->max_leaf_size) {
            node_split_policy curr_node_split_policy;

            /* set init B to -INF, and try to find the max split strategy that maximize B */
            ts_type max_diff_value = (FLT_MAX *(-1));      
            
            
            ts_type avg_children_range_value;
            short hs_split_point = -1;
            short *child_node_points;
            int num_child_node_points;
            const int num_child_segments = 2; //by default split to two subsegments

            // horizontal split
            for (int i = 0; i < node->num_node_points; ++i) {
                segment_sketch curr_node_segment_sketch = node->node_segment_sketches[i];

                //This is the QoS of this segment. QoS is the estimation quality evaluated as =
                //QoS = segment_length * (max_mean_min_mean) * ((max_mean_min_mean) + (max_stdev * max_stdev))
                //The smaller the QoS, the more effective the bounds are for similarity estimation

                ts_type node_range_value = calculate_mean_std_dev_range(curr_node_segment_sketch, get_segment_length(node->node_points, i));

                //for every split policy
                for (int j = 0; j < node->num_node_segment_split_policies; ++j) {
                    struct node_segment_split_policy curr_node_segment_split_policy = node->node_segment_split_policies[j];
                    //to hold the two child segments


                    auto child_node_segment_sketches = static_cast<segment_sketch *>(malloc( sizeof(struct segment_sketch) * num_child_segments));

                    if (child_node_segment_sketches == nullptr) {
                        cerr <<"error in Index.cpp: could not allocate memory for the child node segment sketches for node "
                             << node->filename << endl;
                        return FAILURE;
                    }

                    for (int k = 0; k < num_child_segments; ++k) {
                        child_node_segment_sketches[k].indicators = nullptr;
                        child_node_segment_sketches[k].indicators = static_cast<ts_type *>(malloc(
                                sizeof(ts_type) * curr_node_segment_sketch.num_indicators)); // 子节点的num_indicators和父节点相同
                        if (child_node_segment_sketches[k].indicators == nullptr) {
                            cerr << "Error in Index.cpp: could not allocate memory for the child node segment sketches "
                                    "indicators for node " <<node->filename<<endl;
                            return FAILURE;
                        }

                    }


                    if (is_split_policy_mean(curr_node_segment_split_policy))
                        mean_node_segment_split_policy_split(&curr_node_segment_split_policy,
                                                             curr_node_segment_sketch,
                                                             child_node_segment_sketches);
                    else if (is_split_policy_stdev(curr_node_segment_split_policy))
                        stdev_node_segment_split_policy_split(&curr_node_segment_split_policy,
                                                              curr_node_segment_sketch,
                                                              child_node_segment_sketches);
                    else {
                        cerr << "Error in Index.cpp: Split policy was not set properly for node"<< node->filename<<endl;
                        return FAILURE;
                    }

                    ts_type range_values[num_child_segments];
                    for (int k = 0; k < num_child_segments; ++k) {
                        struct segment_sketch child_node_segment_sketch = child_node_segment_sketches[k];
                        range_values[k] = calculate_mean_std_dev_range(child_node_segment_sketch,
                                                     get_segment_length(node->node_points, i));
                    }

                    //diff_value represents the splitting benefit
                    //B = QoS(N) - (QoS_leftNode + QoS_rightNode)/2
                    //the higher the diff_value, the better is the splitting

                    avg_children_range_value = calc_mean(range_values, 0, num_child_segments);
                    ts_type diff_value = node_range_value - avg_children_range_value;

                    if (diff_value > max_diff_value) {
                        max_diff_value = diff_value;
                        curr_node_split_policy.split_from = get_segment_start(node->node_points, i);
                        curr_node_split_policy.split_to = get_segment_end(node->node_points, i);
                        curr_node_split_policy.indicator_split_idx = curr_node_segment_split_policy.indicator_split_idx;
                        curr_node_split_policy.indicator_split_value = curr_node_segment_split_policy.indicator_split_value;
                        curr_node_split_policy.curr_node_segment_split_policy = curr_node_segment_split_policy;
                    }
                    for (int k = 0; k < num_child_segments; ++k) {
                        free(child_node_segment_sketches[k].indicators);
                    }
                    free(child_node_segment_sketches);
                }
            }

            /* 
             * add trade-off for horizontal split,
             * we bias for minimize number if segments by giving preference to H split more than V split
             */
            max_diff_value = max_diff_value * 2;

            //we want to test every possible split policy for each horizontal segment
            for (int i = 0; i < node->num_hs_node_points; ++i) {
                struct segment_sketch curr_hs_node_segment_sketch = node->hs_node_segment_sketches[i];
                ts_type node_range_value = calculate_mean_std_dev_range(curr_hs_node_segment_sketch,
                                                      get_segment_length(node->hs_node_points, i));

                //for every split policy
                for (int j = 0; j < node->num_node_segment_split_policies; ++j) {
                    struct node_segment_split_policy curr_hs_node_segment_split_policy = node->node_segment_split_policies[j];

                     //to hold the two child segments
                    auto child_node_segment_sketches = static_cast<segment_sketch *>(malloc(sizeof(struct segment_sketch) *
                                                                                       num_child_segments));
                    if (child_node_segment_sketches == nullptr) {
                        cerr <<"Error in INdex.cpp: could not allocate memory \
                               for the horizontal child node segment sketches for \
                               node " << node->filename<<endl;
                        return FAILURE;
                    }

                    for (int k = 0; k < num_child_segments; ++k) {
                        child_node_segment_sketches[k].indicators = nullptr;
                        child_node_segment_sketches[k].indicators = static_cast<ts_type *>(malloc(
                                sizeof(ts_type) * curr_hs_node_segment_sketch.num_indicators));
                        if (child_node_segment_sketches[k].indicators == nullptr) {
                            cerr <<"Error in INdex.cpp: could not allocate memory \
                                   for the horizontal child node segment sketches indicatores for \
                                   node " << node->filename<<endl;
                            return FAILURE;
                        }
                    }

                    if (is_split_policy_mean(curr_hs_node_segment_split_policy))
                        mean_node_segment_split_policy_split(&curr_hs_node_segment_split_policy,
                                                             curr_hs_node_segment_sketch,
                                                             child_node_segment_sketches);
                    else if (is_split_policy_stdev(curr_hs_node_segment_split_policy))
                        stdev_node_segment_split_policy_split(&curr_hs_node_segment_split_policy,
                                                              curr_hs_node_segment_sketch,
                                                              child_node_segment_sketches);
                    else
                        printf("split policy not initialized properly\n");

                    ts_type range_values[num_child_segments];
                    for (int k = 0; k < num_child_segments; ++k) {
                        struct segment_sketch child_node_segment_sketch = child_node_segment_sketches[k];
                        range_values[k] = calculate_mean_std_dev_range(child_node_segment_sketch,
                                                     get_segment_length(node->hs_node_points, i));
                    }

                    avg_children_range_value = calc_mean(range_values, 0, num_child_segments);

                    ts_type diff_value = node_range_value - avg_children_range_value;

                    if (diff_value > max_diff_value) {
                        max_diff_value = diff_value;
                        curr_node_split_policy.split_from = get_segment_start(node->hs_node_points, i);
                        curr_node_split_policy.split_to = get_segment_end(node->hs_node_points, i);
                        curr_node_split_policy.indicator_split_idx = curr_hs_node_segment_split_policy.indicator_split_idx;
                        curr_node_split_policy.indicator_split_value = curr_hs_node_segment_split_policy.indicator_split_value;
                        curr_node_split_policy.curr_node_segment_split_policy = curr_hs_node_segment_split_policy;
                        hs_split_point = get_hs_split_point(node->node_points,
                                                            curr_node_split_policy.split_from,
                                                            curr_node_split_policy.split_to,
                                                            node->num_node_points);
                    }

                    for (int k = 0; k < num_child_segments; ++k) {
                        free(child_node_segment_sketches[k].indicators);
                    }

                    free(child_node_segment_sketches);
                }
            }

            // we create node->split_policy for the choosen split policy
            node->split_policy = nullptr;
            node->split_policy = static_cast<node_split_policy *>(calloc(1, sizeof(struct node_split_policy)));
            if (node->split_policy == nullptr) {
                cerr <<"Error in Index.cpp: could not allocate memory \
                        for the split policy of node  "<< node->filename<<endl;
                return FAILURE;
            }
            node->split_policy->split_from = curr_node_split_policy.split_from;
            node->split_policy->split_to = curr_node_split_policy.split_to;
            node->split_policy->indicator_split_idx = curr_node_split_policy.indicator_split_idx;
            node->split_policy->indicator_split_value = curr_node_split_policy.indicator_split_value;
            node->split_policy->curr_node_segment_split_policy = curr_node_split_policy.curr_node_segment_split_policy;


            //when hs_split_point stays less than 0, it means that
            //considering splitting a vertical segment is not worth it
            //according to the QoS heuristic

            if (hs_split_point < 0) {
                num_child_node_points = node->num_node_points;
                child_node_points = static_cast<short *>(malloc(sizeof(short) * num_child_node_points));
                if (child_node_points == nullptr) {
                   cerr<<"Error in Index.cpp: could not allocate memory "
                         "for the child node segment points node  " <<node->filename<<endl;
                    return FAILURE;
                }
                //children will have the same number of segments as parent
                for (int i = 0; i < num_child_node_points; ++i) {
                    child_node_points[i] = node->node_points[i];
                }
            }
            else {
                num_child_node_points = node->num_node_points + 1;
                child_node_points = static_cast<short *>(malloc(sizeof(short) * num_child_node_points));
                if (child_node_points == nullptr) {
                    cerr <<"Error in Index.c: could not allocate memory for the child node segment points node  "<< node->filename<<endl;
                    return FAILURE;
                }
                //children will have one additional segment than the parent
                for (int i = 0; i < (num_child_node_points - 1); ++i) {
                    child_node_points[i] = node->node_points[i];
                }
                child_node_points[num_child_node_points - 1] = hs_split_point; //initialize newly added point

                qsort(child_node_points, num_child_node_points, sizeof(short),
                      reinterpret_cast<__compar_fn_t>(compare_short));

            }

            //this will put the time series of this node in the file_buffer->buffered_list aray
            //it will include the time series in disk and those in memory

            // 创建node节点左右子树，并且初始化左右子树的segments
            if (!node->split_node(this, child_node_points, num_child_node_points)) {
                fprintf(stderr, "Error in Index.cpp: could not split node %li | %s.\n", node->id,node->filename);
                return FAILURE;
            }

            free(child_node_points);

            node->file_buffer->do_not_flush = true;

            // 检查当前节点是否有文件缓冲区，如果没有则初始化（如果缓冲区不足，则刷新到磁盘）
            if (!node->getOrInitFileBuffer(this)) {
                cerr << "Error in Index.cpp:  Could not get the file buffer for node "<<
                     node->id <<" child of node "<<node->parent->id<<endl;

                return FAILURE;
            }

            //copying the contents of the the node being split in case it gets flushed from memory to disk
            VectorWithIndex *ts_list = node->getTS(this);
            
            // 🛡️ CRITICAL FIX: 添加空指针检查，防止程序崩溃
            if (ts_list == nullptr) {
                std::cerr << "[FATAL ERROR] Failed to get time series data from node " << node->id 
                          << ". Memory allocation failed. Cannot proceed with node splitting." << std::endl;
                return false; // 或者抛出异常/返回错误码
            }

            cout <<"[SPLITTING] Leaf Node "<< node->id
                 <<", of node size "<< node->node_size << ", and disk size "<<node->file_buffer->disk_count
            << ", will be splitted into 2 Leaf node!"<<endl;


            // for (int idx = 0; idx < this->index_setting->max_leaf_size; ++idx) {
            for (int idx = 0; idx < node->node_size; ++idx) {
                if (node->node_split_policy_route_to_left(ts_list[idx].ts)) {

                    node->left_child->updateStatistics(ts_list[idx].ts);
                    node->left_child->addTS(this,&ts_list[idx], true);

                } else {

                    node->right_child->updateStatistics(ts_list[idx].ts);
                    node->right_child->addTS(this,&ts_list[idx], true);
                }
            }

            for (int i = 0; i < node->node_size; i++){
                free(ts_list[i].ts);
                ts_list[i].ts = nullptr;

                free(ts_list[i].ts_index);
                ts_list[i].ts_index = nullptr;
            }
            free(ts_list);

            cout <<"[splitting success] node "
                 << node->id<<" has been splitted into 2 new leaves of size "<<node->left_child->node_size
                 <<" and "<<node->right_child->node_size
                 <<endl;

            // 当前节点已经分裂完成，可以删除node文件缓冲区和文件缓冲区映射
            if(!node->deleteFileBuffer(this)){
                cerr <<"error in hercules_index.c: could not delete file buffer for node "<< node->filename<<endl;
                return FAILURE;
            };
        }
    }
    return SUCCESS;
}


/*
 * add Node's File Buffer(node->file_buffer) to the file buffer map(index->buffer_manager), if the map doesnt exist we init it
 */
bool Index::addFileBuffer(Node *node) const {


    unsigned int idx = this->buffer_manager->file_map_size; //  buffer_manager->file_map_size ||  buffer_manager->file_map

    if (idx == 0) {
        this->buffer_manager->file_map = static_cast<hercules_file_map *>(malloc(sizeof( hercules_file_map)));
        if (this->buffer_manager->file_map == nullptr) {
            fprintf(stderr, "Error in Index.c: Could not allocate memory for the file map.\n");
            return FAILURE;
        }
        this->buffer_manager->file_map[idx].file_buffer = node->file_buffer;
        node->file_buffer->position_in_map = &this->buffer_manager->file_map[idx];

        //index->buffer_manager->file_map[idx].filename = node->filename;
        this->buffer_manager->file_map[idx].prev = nullptr;
        this->buffer_manager->file_map[idx].next = nullptr;

        this->buffer_manager->file_map_tail = this->buffer_manager->file_map; // 头节点 即 尾结点

    } else {
        struct hercules_file_map *lastP = this->buffer_manager->file_map_tail;
        lastP->next = static_cast<hercules_file_map *>(malloc(sizeof(hercules_file_map)));

        if (lastP->next == nullptr) {
            fprintf(stderr, "Error in Index.c: Could not allocate memory for new entry in the Buffer files map.\n");
            return FAILURE;
        }

        lastP->next->file_buffer = node->file_buffer;
        node->file_buffer->position_in_map = lastP->next;

        lastP->next->prev = lastP;
        lastP->next->next = nullptr;
        this->buffer_manager->file_map_tail = lastP->next;

    }

    ++this->buffer_manager->file_map_size;

    return SUCCESS;
}

char * Index::getIndexFilename() const{
    char *filename = static_cast<char *>(malloc(sizeof(char)* (strlen(this->index_setting->index_path_hercules) + 9)));
    filename = strcpy(filename, this->index_setting->index_path_hercules);
    filename = strcat(filename, "root.idx\0");
    return filename;
}

void Index::write() {

    cout<< "[Storing Index] Store hercules tree in "<< this->index_setting->index_path_hercules<<endl;
    char* filename = this->getIndexFilename();
    std::cout << "filename: " << filename << std::endl;

    FILE *file = fopen(filename, "wb");
    free(filename);
    if (file == nullptr) {
        cerr <<"Error in Index.cpp: Could not open the index file. Reason = "<< strerror(errno)<<endl;
        exit(-1);
    }

    unsigned int timeseries_size = this->index_setting->timeseries_size;
    unsigned int max_leaf_size = this->index_setting->max_leaf_size;
    unsigned int init_segments = this->index_setting->init_segments;
    double buffered_memory_size = this->index_setting->buffered_memory_size;

    /* settings data */
    fwrite(&Node::num_leaf_node, sizeof(unsigned long), 1, file);
    fwrite(&buffered_memory_size, sizeof(double), 1, file);
    fwrite(&timeseries_size, sizeof(unsigned int), 1, file);
    fwrite(&init_segments, sizeof(unsigned int), 1, file);
    fwrite(&max_leaf_size, sizeof(unsigned int), 1, file);


    Node::num_leaf_node = 0;

    /* nodes and file buffer */
    this->first_node->write(this, file);

    fclose(file);

    cout << "[hercules tree Storing Finished]" << endl;
}


/* path： */
Index *Index::Read(char *path, unsigned int mode) {
    return new Index(path, mode);
}


/* load inde from root_directory */
Index::Index(char *root_directory, unsigned int mode) {
    this->in_memory = mode-1;
    this->time_stats = new timelapse ;
    this->time_stats->index_building_time = 0;
    this->time_stats->querying_time = 0;
    this->index_path = root_directory;
    if (access(root_directory, F_OK) != 0) {
       fprintf(stderr, "the directory does not exist. Please provide a valid directory.\n");
       exit(-1);
    }

    char *index_path_hercules = (char *) malloc_index(sizeof(char) * (strlen(root_directory) + 7));
    index_path_hercules = strcpy(index_path_hercules, root_directory);
    index_path_hercules = strcat(index_path_hercules, "tree/\0");

    char *index_path_hnsw = (char *) malloc_index(sizeof(char) * (strlen(root_directory) + 8));
    index_path_hnsw = strcpy(index_path_hnsw, root_directory);
    index_path_hnsw = strcat(index_path_hnsw, "graph/\0");

    char *index_path_txt = (char *) malloc_index(sizeof(char) * (strlen(root_directory) + 6));
    index_path_txt = strcpy(index_path_txt, root_directory);
    index_path_txt = strcat(index_path_txt, "txt/\0");
    
    cout<< "[loading hercules tree] path : " << index_path_hercules << endl;
    char *tree_full_filename = (char *) malloc_index(sizeof(char) * (strlen(index_path_hercules) + 9));
    tree_full_filename = strcpy(tree_full_filename, index_path_hercules);
    tree_full_filename = strcat(tree_full_filename, "root.idx\0");
    cout<<"tree_full_filename:"<<tree_full_filename<<endl;

    char cwd[1024];
    if(getcwd(cwd, sizeof(cwd)) != NULL) {
        cout << "current working directory: " << cwd << endl;
    }


    FILE *file = fopen(tree_full_filename, "rb");
    if(file == nullptr) {
        cerr << "Error opening file: " << strerror(errno) << endl;
        exit(-1);
    }
    free(tree_full_filename);


    // SETTINGS DATA
    unsigned long count_leaves = 0;
    unsigned int timeseries_size = 0;
    unsigned int max_leaf_size = 0;
    unsigned int init_segments = 0;
    double buffered_memory_size = 0;

    fread(&count_leaves, sizeof(unsigned long), 1, file);
    fread(&buffered_memory_size, sizeof(double), 1, file);
    fread(&timeseries_size, sizeof(unsigned int), 1, file);
    fread(&init_segments, sizeof(unsigned int), 1, file);
    fread(&max_leaf_size, sizeof(unsigned int), 1, file);



    /* ///////////////////////////////////////////////////////////////////////////////// */
    // const char *index_path_hercules, const char *index_path_hnsw, const char *index_path_txt,  const char *index_path
    this -> index_setting = new Setting(index_path_hercules, index_path_hnsw, index_path_txt, root_directory, timeseries_size, init_segments,
                                        max_leaf_size, buffered_memory_size, -1, -1);


    this -> buffer_manager = nullptr;  //we dont use mem_array while querying

    this -> l2space = new hnswlib::L2Space(index_setting->timeseries_size);

    this -> first_node = Node::Read(this, file, mode);


    if(count_leaves != Node::num_leaf_node){
        cerr << "[Bug]Number of expected leaves : "<<count_leaves
        <<", Number of loaded leaves : "<<Node::num_leaf_node<<endl;
        exit(1);
    }

    fclose(file);

    fprintf(stdout, "[Index Loaded Successfully] from: %s "
                    "||  Number of Leaf Node : %lu "
                    "||  NUmber of Internal Node : %lu \n", root_directory, Node::num_leaf_node, Node::num_internal_node);


}

Index::~Index(){
    delete this->time_stats;
    delete this->index_setting;
    delete first_node; 
    delete this -> l2space;  ////// Valgrind
}

Node **Index::getLeaves(int &i) {
    Node ** leaves = static_cast<Node **>(malloc(sizeof(Node *) * this->first_node->num_leaf_node));
    first_node->getLeaves(leaves,i);
    return leaves;
}

Node **Index::getLeaves() {
    Node ** leaves = static_cast<Node **>(malloc(sizeof(Node *) * this->first_node->num_leaf_node));
    int i=0;
    first_node->getLeaves(leaves,i);
    return leaves;
}


/**
 <center>
 <b>copy append node buffered_list to disk file, <br>
 and increment the disk_count for each ts. <br>
 turn node.in_disk to true and clear filebuffer</b></center

 * */

