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
	Index *index;

	Node **leaves;
	int num_leaf_node;
	int leaf_size;

	unordered_map<file_position_type, Node *> ts_leaf_map; // Using string as key for vector<ts_type> serialization
	unordered_map<int, int> leafId2Idx;
	std::vector<std::vector<int>> leaf_topk_indices; // leaf id -> top-k indices mapping

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

	leaf_centroid *centroids;

	const char* statistics_path="/home/xln/elpis/data/statistics/";

	Hercules(char *dataset, int dataset_size, char *index_path, int timeseries_size, int leaf_size,
			 char *query_dataset, int query_dataset_size,
			 char *groundtruth_dataset, int groundtruth_dataset_size,int groundtruth_top_k)
	{


		this->index_path = index_path;
		this->leaf_size=leaf_size;
		this->timeseries_size = timeseries_size;
		this->index = Index::initIndex(this->index_path, this->timeseries_size, 512 * 1024, 1, this->leaf_size, 500, 4); // 参数暂时不用

		// base dataset
		this->dataset = dataset;
		this->dataset_size = dataset_size;
		std::ifstream in(this->dataset, std::ios::binary);
		this->ts_list = new VectorWithIndex[this->dataset_size];
		for (size_t i = 0; i < this->dataset_size; i++)
		{
			this->ts_list[i].ts_index = new file_position_type;	
			this->ts_list[i].ts = (ts_type*)malloc_index(sizeof(ts_type) * timeseries_size);
		}
		
		for (int i = 0; i < this->dataset_size; i++)
		{
			*(this->ts_list[i].ts_index)=i;
			in.read(reinterpret_cast<char *>(this->ts_list[i].ts), this->timeseries_size * sizeof(ts_type));
		}

		// groundtruth dataset
		if (groundtruth_dataset != nullptr && groundtruth_dataset_size > 0)
		{
			this->groundtruth_dataset = groundtruth_dataset;
			this->groundtruth_dataset_size = groundtruth_dataset_size;
			this->groundtruth_top_k=groundtruth_top_k;
			this->groundtruth_list = new int *[this->groundtruth_dataset_size];

			std::ifstream groundtruth_in(this->groundtruth_dataset, std::ios::binary);
			for (int i = 0; i < groundtruth_dataset_size; i++)
			{
				this->groundtruth_list[i] = new int[this->groundtruth_top_k];
				groundtruth_in.read(reinterpret_cast<char *>(this->groundtruth_list[i]), this->groundtruth_top_k * sizeof(int));
			}
		}

	}

	/* ******************************* 建立索引树************************************** */
	void buildIndexTree()
	{
		this->index->buildIndexFromBinaryData(this->dataset, this->dataset_size);
		this->index->write();
		this->num_leaf_node = this->index->first_node->num_leaf_node;
		cout<<"Number of leaf nodes: "<<this->num_leaf_node<<endl;
		cout<<"Number of internal nodes: "<<this->index->first_node->num_internal_node<<endl;

		cout<<"before this->index->getLeaves()"<<endl;
		this->leaves = this->index->getLeaves();
		cout<<"after this->index->getLeaves()"<<endl;

	}


	/* ******************************* 叶子节点id映射************************************** */
	void generateLeafId2IdxMap(const char* output_path=nullptr)
	{
		leafIndexToIdMap();
		writeLeafId2IdxFile(output_path);
	}

	//  建立叶子节点id到下标的映射，变成连续的数组
	void leafIndexToIdMap(){
		for(int i=0;i<num_leaf_node;i++){
			Node* leaf=leaves[i];
			leafId2Idx[leaf->id]=i;
		}
	}

	// 将leafId2Idx写入文件
	void writeLeafId2IdxFile(const char* output_path_) {
		char* output_path;
		char * leafId2Idx_path = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_hnsw) + strlen("leafId2Idx.txt"))));
		leafId2Idx_path = strcpy(leafId2Idx_path, this->index->index_setting->index_path_hnsw);
		leafId2Idx_path = strcat(leafId2Idx_path, "leafId2Idx.txt");
		if (output_path_ == nullptr || strlen(output_path_) == 0) {
			output_path = leafId2Idx_path;
		} else {
			output_path = const_cast<char*>(output_path_);
		}
		std::ofstream ofs(output_path);
		if (!ofs.is_open()) {
			std::cerr << "can not open file: " << output_path << std::endl;
			return;
		}
		for (const auto& pair : leafId2Idx) {
			ofs << pair.first << " " << pair.second << "\n";
		}
		std::cout << "leafId2Idx.txt has been written successfully." << std::endl;
	}

	/* ******************************* 生成叶子节点文件************************************** */
	// 生成segment
	ts_type* generate_segment(Node* node, size_t* segments_dimension){

        int segment_number=node->num_node_points;
        ts_type* segments=static_cast<ts_type *>(malloc_index(sizeof(ts_type) *4 * segment_number));
        
        for(size_t i=0;i<segment_number;i++){
            segments[4*i+0]=node->node_segment_sketches[i].indicators[0];
            segments[4*i+1]=node->node_segment_sketches[i].indicators[1];
            segments[4*i+2]=node->node_segment_sketches[i].indicators[2];
            segments[4*i+3]=node->node_segment_sketches[i].indicators[3];
        }
        *segments_dimension=segment_number*4;
        return segments;
    }
	// 创建叶子节点文件
	void generate_leafnode_file(){
		for(int i=0;i<num_leaf_node;i++){
            Node* leaf=leaves[i];
            VectorWithIndex *rec = leaf->getTS(index);
			if(rec==nullptr){
				cout<<"加载向量失败"<<endl;
				exit(-1);
			}
            int rec_vecs_number=leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;
    
            // 设置当前节点向量所在文件名称
            char *index_full_filename = leaf->setLeafFileName(index); 

            // 写入数据到文本文件
            std::ofstream ofs(index_full_filename);
            if (!ofs) {
                std::cerr << "无法打开文件: " << index_full_filename << std::endl;
                continue;
            }
			// 最先写入叶子节点的id
			ofs << "Leaf Node ID: " << leaf->id << "\n";

            // 第一行写入segment文件
            size_t seg_dim;
            ts_type* node_segment = generate_segment(leaf, &seg_dim);
            if (node_segment == nullptr) {
                std::cerr << "Segment generation failed for node " << leaf->id << std::endl;
                continue;
            }
            for (size_t j = 0; j < seg_dim; ++j) {
                ofs << node_segment[j];
                if (j != seg_dim - 1) ofs << " ";
            }
            ofs << "\n";
            free(node_segment);

                
            // 向量rec[i][j] 二维数组，
            for (int r = 0; r < rec_vecs_number; ++r) {
                for (int c = 0; c < this->timeseries_size; ++c) {
                    ofs << rec[r].ts[c];
                    if (c != this->timeseries_size - 1) ofs << " ";
                }
                ofs << "\n";
            }
            ofs.close();
            std::cout <<"Write finished: " << index_full_filename << std::endl;
            free(index_full_filename);
        }
	}


	/* **********************生成叶子节点质心，并按照映射生成最终的二维质心数组***************************** */
	void generate_leaf_centroids(const char* output_path=nullptr, const char* generate_way = "Centroid"){

		// 所有叶节点的质心
		if (this->num_leaf_node <= 0) {
			std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
			return;
		}

		leaf_centroid* centroids_temp=(leaf_centroid*)malloc_index(sizeof(leaf_centroid) * this->num_leaf_node);
		for (int i = 0; i < this->num_leaf_node; i++) {
			Node *leaf = this->leaves[i]; 
			int rec_vecs_number = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count; // 获取叶子包含的向量数目
			centroids_temp[i].ts_centroid = static_cast<ts_type *>(malloc_index(sizeof(ts_type) * this->timeseries_size));
			centroids_temp[i].leaf_centroid_index = static_cast<file_position_type *>(malloc_index(sizeof(file_position_type)));
			memset(centroids_temp[i].ts_centroid, 0, sizeof(ts_type) * this->timeseries_size);  // 初始化为0
			*(centroids_temp[i].leaf_centroid_index) = leaf->id; // 将叶子节点的id存入质心索引

			VectorWithIndex *rec = leaf->getTS(index);
			if (rec == nullptr) {
				std::cerr << "Error: Failed to get time series from leaf node." << std::endl;
				exit(-1);
			}

			// 遍历每个点，计算到所有其他点的距离，并选择最小的点
			if(strcmp(generate_way, "Center") == 0){ 
				file_position_type center_index=0;
				double min_distance = numeric_limits<double>::infinity();  // 最小距离初始化为无穷大
                for (int j = 0; j < rec_vecs_number; j++) {
                    double total_distance = 0.0;
                    for (int k = 0; k < rec_vecs_number; k++) {
                        if (j != k) {
                         // 计算 rec_vecs[j] 到 rec_vecs[k] 的欧氏距离
                         total_distance += euclideanDistance(rec[j].ts, rec[k].ts,this->timeseries_size);
                        }
                    }

                    // 记录当前点的总距离，选择总距离最小的点作为中心
                    if (total_distance < min_distance) {
                        min_distance = total_distance;
                        center_index = j;
                    }
                }
				for(int j=0;j<this->timeseries_size;j++){
				   centroids_temp[i].ts_centroid[j]=rec[center_index].ts[j];
				}

			}else if(strcmp(generate_way, "Centroid") == 0){ // 质心
			    for (int j = 0; j < rec_vecs_number; j++) {
			    	for (int k = 0; k < this->timeseries_size; k++) {
			    		centroids_temp[i].ts_centroid[k] += rec[j].ts[k];
			    	}
			    }
    
			    for (int k = 0; k < this->timeseries_size; k++) {
			    	centroids_temp[i].ts_centroid[k] /= rec_vecs_number;
			    }
			}

			for (int j = 0; j < rec_vecs_number; j++) {
                free(rec[j].ts);
                free(rec[j].ts_index);
            }
			free(rec);
		}
		// 将质心位置映射和label一致
		this->centroids = static_cast<leaf_centroid *>(malloc_index(sizeof(leaf_centroid) * this->num_leaf_node));
		for(int i = 0; i < this->num_leaf_node; i++) {
			this->centroids[i].ts_centroid = static_cast<ts_type *>(malloc_index(sizeof(ts_type) * this->timeseries_size));
			this->centroids[i].leaf_centroid_index = static_cast<file_position_type *>(malloc_index(sizeof(file_position_type)));
			*(this->centroids[i].leaf_centroid_index) = 0; // 初始化为0，后续会被覆盖
		}

		for(int i = 0; i < this->num_leaf_node; i++) {
			int idx_leaf=*(centroids_temp[i].leaf_centroid_index);
			int idx = this->leafId2Idx[idx_leaf];
			for(int j=0;j<this->timeseries_size;j++){
				this->centroids[idx].ts_centroid[j] = centroids_temp[i].ts_centroid[j];
			}
			*(this->centroids[idx].leaf_centroid_index) = *(centroids_temp[i].leaf_centroid_index);
		}

		// 写入质心到文件
		char *leaf_centroids_path = static_cast<char *>(malloc_index(sizeof(char) * (strlen(this->index->index_setting->index_path_hnsw) + strlen("leaf_")+strlen(generate_way)+strlen(".txt")+1)));
		leaf_centroids_path = strcpy(leaf_centroids_path, this->index->index_setting->index_path_hnsw);
		leaf_centroids_path = strcat(leaf_centroids_path, "leaf_");
		leaf_centroids_path = strcat(leaf_centroids_path, generate_way);
		leaf_centroids_path = strcat(leaf_centroids_path, ".txt");
		
		char* output_file = (output_path == nullptr || strlen(output_path) == 0) ? leaf_centroids_path : const_cast<char*>(output_path);
		std::ofstream ofs(output_file);
		if (!ofs.is_open()) {
			std::cerr << "can not open file: " << output_file << std::endl;
			exit(-1);
		}
		// 按照label中叶子节点的顺序将质心写入文件
		for (int i = 0; i < this->num_leaf_node; i++) { // leafId2Idx[leaf->id]
			// ofs << "Leaf Node ID: " << *(this->centroids[i].leaf_centroid_index) << "\n";
			for (int j = 0; j < this->timeseries_size; j++) {
				ofs << std::fixed << this->centroids[i].ts_centroid[j];
				if (j != this->timeseries_size - 1) ofs << " ";
			}
			ofs << "\n";
		}
		std::cout << "Leaf centroids have been written to " << output_file << std::endl;
		free(leaf_centroids_path);
		for (int i = 0; i < this->num_leaf_node; i++) {
			free(centroids_temp[i].ts_centroid);
			free(centroids_temp[i].leaf_centroid_index);
		}
		free(centroids_temp);
	}

    /* ******************************* 建立时间序列id到叶子节点的映射，便于生成label************************************** */
 	void generate_ts_leaf_map_file(const char* output_path=nullptr){
		fillTsLeafMap();
		writeTsLeafMapFile(output_path);
	}


    // 填充哈希表
	void fillTsLeafMap()
	{
		for (int i = 0; i < this->num_leaf_node; i++)
		{
			Node *leaf = this->leaves[i];
			VectorWithIndex *rec = leaf->getTS(index);
			int rec_vecs_number=leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;
			for (int j = 0; j < rec_vecs_number; j++)
			{
				file_position_type ts_key=*(rec[j].ts_index);
				auto result = this->ts_leaf_map.insert({ts_key, leaf});
                if (!result.second) {
                    // 插入失败，说明键已经存在
                    std::cerr << "Key already exists: " << ts_key << std::endl;
                }

			}
			for(int i=0;i<rec_vecs_number;i++){
				free(rec[i].ts);
				free(rec[i].ts_index);
			}
			free(rec);
		}
	}

	void writeTsLeafMapFile(const char* output_path_)
	{

		char* output_path;
		char * ts_leaf_map = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_hnsw) + strlen("/ts_leaf_map.txt"))));
		ts_leaf_map = strcpy( ts_leaf_map , this->index->index_setting->index_path_hnsw);
        ts_leaf_map  = strcat( ts_leaf_map , "/ts_leaf_map.txt");
		if(output_path_ == nullptr || strlen(output_path_) == 0){
			output_path = ts_leaf_map;
		}else{
			output_path = const_cast<char*>(output_path_);
		}

		std::ofstream out(output_path, std::ios::app);
		for (unordered_map<file_position_type, Node *>::iterator it = this->ts_leaf_map.begin(); it != this->ts_leaf_map.end(); ++it)
		{
			file_position_type ts_key = it->first;
			Node *leaf = it->second;
			// out << "Leaf Node: " << leaf->id << " at " << leaf << ", " << "The leaf filename is " << leaf->filename << ". ";
			out<<"ts_key: "<<ts_key;
			out<<" leaf->id: "<<leaf->id;
			out<< endl;
		}
		std::cout << "ts_leaf_map.txt has been written successfully." << std::endl;
	}

	/* ******************************* top-k----> leaf-id *************************** */
	void topk2LeafId(const char* output_path=nullptr)
	{
		knn_groundtruth = new file_position_type *[this->groundtruth_dataset_size];

		for (int i = 0; i < this->groundtruth_dataset_size; i++)
		{
			knn_groundtruth[i] = new file_position_type[this->groundtruth_top_k];
			memset(knn_groundtruth[i], 0, this->groundtruth_top_k * sizeof(file_position_type));
			for (int j = 0; j < this->groundtruth_top_k; j++)
			{
				int ts_key=this->groundtruth_list[i][j];
				if(ts_key>=this->dataset_size || ts_key<0){
					cout<<"ts_key="<<ts_key<<"is not between[0,"<<this->dataset_size<<"]"<<endl;
					exit(-1);
				}

				auto it = this->ts_leaf_map.find(ts_key);
				Node *leaf=nullptr;
                if (it != this->ts_leaf_map.end()) {
                    leaf = it->second;
					if (leaf == nullptr) {
                       std::cerr << "Error: Node* for key " << ts_key << " is nullptr!" << std::endl;
                       exit(-1);
                   }
                } else{
				    std::cerr << "Error: Key not found: " << ts_key << std::endl;
					exit(-1);
				}
                knn_groundtruth[i][j]= leaf->id; // 将叶子节点的id存入knn_groundtruth[i][j]

			}
		}

	    // 输出knn_groundtruth的内容到文件中
		char * knn_groundtruth_path = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_hnsw) + strlen("knn_groundtruth.txt")+1)));
		knn_groundtruth_path = strcpy(knn_groundtruth_path, this->index->index_setting->index_path_hnsw);
		knn_groundtruth_path = strcat(knn_groundtruth_path, "knn_groundtruth.txt");
		if (output_path == nullptr || strlen(output_path) == 0) {
			output_path = knn_groundtruth_path;
		} else {
			output_path = const_cast<char*>(output_path);
		}

		std::ofstream ofs(output_path);
		if (!ofs.is_open()) {
			std::cerr << "can not open file: " << output_path << std::endl;
			exit(-1);
		}
		for (int i = 0; i < this->groundtruth_dataset_size; ++i) {
			for (int j = 0; j < this->groundtruth_top_k; ++j) {
				ofs << this->knn_groundtruth[i][j];
				if (j != this->groundtruth_top_k - 1) ofs << " ";
			}
			ofs << "\n";
		}


	}

	void leafContainsTopK(int selected_k)
{

	// 创建一个文件，保存每个叶子节点的top-k下标
    std::ostringstream oss;
    oss << this->statistics_path<<"/"<<getFileNameWithoutBin(this->dataset) <<"/"<< "leafContainsTop" << selected_k <<"_leafsize"<<this->index->index_setting->max_leaf_size<< ".txt";
    std::string leafContainsTopK_path_str = oss.str();
	cout<<"leafContainsTopK_path_str="<<leafContainsTopK_path_str<<endl;
    std::ofstream ofs(leafContainsTopK_path_str.c_str());

    std::ostringstream oss_sta;
    oss_sta << this->statistics_path <<"/"<<getFileNameWithoutBin(this->dataset) <<"/"<< "leafContainsTop" << selected_k << "_leafsize"<<this->index->index_setting->max_leaf_size<< "_sta.txt";
    std::string leafContainsTopK_path_str_sta = oss_sta.str();
    std::ofstream ofs_sta(leafContainsTopK_path_str_sta.c_str());

    for (int i = 0; i < this->groundtruth_dataset_size; i++)
    {
	    // 初始化每个叶子节点的top-k下标统计
        leaf_topk_indices.clear();
        leaf_topk_indices.resize(this->num_leaf_node);
        for (int j = 0; j < selected_k; j++)
        {
            int ts_key = this->groundtruth_list[i][j];
            if (ts_key >= this->dataset_size || ts_key < 0) {
                cout << "ts_key=" << ts_key << "is not between[0," << this->dataset_size << "]" << endl;
                exit(-1);
            }

            auto it = this->ts_leaf_map.find(ts_key);
            Node *leaf = nullptr;
            if (it != this->ts_leaf_map.end()) {
                leaf = it->second;
                if (leaf == nullptr) {
                    std::cerr << "Error: Node* for key " << ts_key << " is nullptr!" << std::endl;
                    exit(-1);
                }
            } else {
                std::cerr << "Error: Key not found: " << ts_key << std::endl;
                exit(-1);
            }
            // 用映射找到叶子节点在数组中的下标
            auto idx_it = leafId2Idx.find(leaf->id);
            if (idx_it == leafId2Idx.end()) {
                std::cerr << "Error: leaf->id " << leaf->id << " not found in leafId2Idx map!" << std::endl;
                exit(-1);
            }
            int leaf_idx = idx_it->second;
            // 记录该query的top-k下标属于哪个叶子
            leaf_topk_indices[leaf_idx].push_back(j);
        }


        ofs << "query " << i << "*************************"<<endl;
        for (int leaf_idx = 0; leaf_idx < this->num_leaf_node; ++leaf_idx) {
            if (!leaf_topk_indices[leaf_idx].empty()) {
                ofs << "Leaf " << leaf_idx <<" : ";
                for (int idx : leaf_topk_indices[leaf_idx]) {
                    ofs << idx << " ";
                }
                ofs << endl;
            }
        }

		// ofs_sta<< "query " << i << "*************************"<<endl;
		int count=0;
		for (int leaf_idx = 0; leaf_idx < this->num_leaf_node; ++leaf_idx) {
            if (!leaf_topk_indices[leaf_idx].empty()) {
				count++;
            }
        }
		ofs_sta<<count<<" "<<endl;
    }

	ofs.close();
	ofs_sta.close();
	}


	/* ******************************* 生成标签************************************** */
	void generate_label(const char* output_path=nullptr){
		calcKNNinLeaves();
		writeKNNDistributionsToFile(output_path);
	}
	
	// 生成label：每个query的top-k在叶子中的分布
	void calcKNNinLeaves()
	{
		knn_distributions = new file_position_type *[this->groundtruth_dataset_size];

		for (int i = 0; i < this->groundtruth_dataset_size; i++)
		{
			knn_distributions[i] = new file_position_type[this->num_leaf_node];
			memset(knn_distributions[i], 0, this->num_leaf_node * sizeof(file_position_type));
			for (int j = 0; j < this->groundtruth_top_k; j++)
			{
				int ts_key=this->groundtruth_list[i][j];
				if(ts_key>=this->dataset_size || ts_key<0){
					cout<<"ts_key="<<ts_key<<"is not between[0,"<<this->dataset_size<<"]"<<endl;
					exit(-1);
				}

				auto it = this->ts_leaf_map.find(ts_key);
				Node *leaf=nullptr;
                if (it != this->ts_leaf_map.end()) {
                    leaf = it->second;
					if (leaf == nullptr) {
                       std::cerr << "Error: Node* for key " << ts_key << " is nullptr!" << std::endl;
                       exit(-1);
                   }
                } else{
				    std::cerr << "Error: Key not found: " << ts_key << std::endl;
					exit(-1);
				}
                // 用映射找到叶子节点在数组中的下标
                auto idx_it = leafId2Idx.find(leaf->id);
                if (idx_it == leafId2Idx.end()) {
                    std::cerr << "Error: leaf->id " << leaf->id << " not found in leafId2Idx map!" << std::endl;
                    exit(-1);
                }
                int leaf_idx = idx_it->second;
                ++knn_distributions[i][leaf_idx];
			}

			// 检查knn_distribution[i]所有元素的和是否为groundtruth_top_k
			file_position_type sum = 0;
			for (int k = 0; k < this->num_leaf_node; k++) {
				sum += knn_distributions[i][k];
			}
			if (sum != this->groundtruth_top_k) {
				cout<<"knn_distribution["<<i<<"] sum" << sum<<"!= groundtruth_top_k"<<endl;
				exit(-1);
			}

			// 输出knn_distributions[i]的内容
			cout<<"knn_distributions["<<i<<"]: ";
			for(int k=0;k<this->num_leaf_node;k++){
				cout<<knn_distributions[i][k]<<" ";
			}
			cout<<endl;
		}
	}


    void writeKNNDistributionsToFile(const char* output_path_) {

		char* output_path;
		char * knn_distribution_path = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_hnsw) + strlen("knn_distributions.txt"))));
		knn_distribution_path = strcpy(knn_distribution_path, this->index->index_setting->index_path_hnsw);	
		knn_distribution_path = strcat(knn_distribution_path, "knn_distributions.txt");
		if (output_path_ == nullptr || strlen(output_path_) == 0) {
			output_path = knn_distribution_path;
		} else {
			output_path = const_cast<char*>(output_path_);
		}
        std::ofstream ofs(output_path);
        if (!ofs.is_open()) {
            std::cerr << "can not open file: " << output_path << std::endl;
            exit(-1);
        }
    
        for (int i = 0; i < this->groundtruth_dataset_size; ++i) {
            for (int j = 0; j < this->num_leaf_node; ++j) {
                ofs << this->knn_distributions[i][j];
                if (j != this->num_leaf_node - 1) ofs << " ";
            }
            ofs << "\n";
        }
    
        ofs.close();
        std::cout << "写入完成：" << output_path << std::endl;
    }


	// 析构函数
	~Hercules() {
		if (this->index) {
			delete this->index;
		}
		if (this->ts_list) {
			for (int i = 0; i < this->dataset_size; i++) {
				free(this->ts_list[i].ts);
				free(this->ts_list[i].ts_index);
			}
			free(this->ts_list);
		}
		if (this->groundtruth_list) {
			for (int i = 0; i < this->groundtruth_dataset_size; i++) {
				free(this->groundtruth_list[i]);
			}
			free(this->groundtruth_list);
		}
		if (this->leaves) {
			free(this->leaves);
		}
		if (this->centroids) {
			for (int i = 0; i < this->num_leaf_node; i++) {
				free(centroids[i].ts_centroid);
				free(centroids[i].leaf_centroid_index);
			}
			free(centroids);
		}
		if (this->knn_distributions) {
			for (int i = 0; i < this->groundtruth_dataset_size; i++) {
				free(this->knn_distributions[i]);
			}
			delete[] this->knn_distributions;
		}
		if(this->knn_groundtruth) {
			for (int i = 0; i < this->groundtruth_dataset_size; i++) {
				free(this->knn_groundtruth[i]);
			}
			delete[] this->knn_groundtruth;
		}
	}
};
