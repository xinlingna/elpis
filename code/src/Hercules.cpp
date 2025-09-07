#include "Hercules.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <limits>


    Hercules::Hercules(char *dataset, unsigned int dataset_size,
			 char *query_dataset, unsigned int query_dataset_size,
			 char *groundtruth_dataset, unsigned int groundtruth_dataset_size,unsigned int groundtruth_top_k, 
			 char * learn_dataset, unsigned int learn_dataset_size, char * learn_groundtruth_dataset,
			 char *index_path, unsigned int timeseries_size, unsigned int leaf_size,
             unsigned int nprobes, bool parallel, unsigned int nworker, bool flatt,
			 int efConstruction, unsigned int m, int efSearch, unsigned int k, unsigned int ep,
			 char *model_file, float zero_edge_pass_ratio, bool search_withWeight, float μ, float T, float thres_probability, int mode)
	{
		/* 
		 * nprobes：搜索的叶子节点限制
		 * parallel：是否使用并行
		 * nworker：搜索叶子节点的工作线程数量
		 * flatt：是否扁平化
		 * construction：构建HNSW的参数
		 * m：每个节点的最大连接数
		 * k：查询时返回的最近邻数量
		 * 
		 * 
		 * ep：训练权重轮次
		 * zero_edge_pass_ratio：0边权通过比例
		 */


		// hercules tree param
		this->index_path = index_path;
		this->leaf_size=leaf_size;
		this->timeseries_size = timeseries_size;
		this->index = Index::initIndex(this->index_path, this->timeseries_size, 3072 * 1024, 1, this->leaf_size, efConstruction, m); // 参数暂时不用
    	Node::max_leaf_size = this->leaf_size;

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


		// edge weight param
 		this->mode=mode; 
		this->query_dataset=query_dataset; // 检验权重训练的结果
		this->query_dataset_size=query_dataset_size;
		this->learn_dataset=learn_dataset; // 训练权重
		this->learn_dataset_size=learn_dataset_size;
		this->learn_groundtruth_dataset=learn_groundtruth_dataset;

		this->nprobes=nprobes;
		this->parallel=parallel;
		this->nworker=nworker;
		this->flatt=flatt;
		this->k=k;
		this->efSearch=efSearch;
		this->ep=ep;
		this->model_file=model_file;
		this->zero_edge_pass_ratio=zero_edge_pass_ratio;
		this->search_withWeight=search_withWeight;
		this->μ=μ;
		this->T=T;
		this->thres_probability=thres_probability;

		// learn_groundtruth_list
		if (learn_groundtruth_dataset != nullptr && learn_dataset_size > 0)
		{
			this->learn_groundtruth_list = new int *[this->learn_dataset_size];

			std::ifstream learn_groundtruth_in(this->learn_groundtruth_dataset, std::ios::binary);
			for (int i = 0; i < this->learn_dataset_size; i++)
			{
				this->learn_groundtruth_list[i] = new int[this->groundtruth_top_k];
				learn_groundtruth_in.read(reinterpret_cast<char *>(this->learn_groundtruth_list[i]), this->groundtruth_top_k * sizeof(int));
			}
		}

	}

	/* ******************************* 建立索引树 保存hercules tree and graph************************************** */
    void Hercules::buildIndexTree()
	{
		this->index->buildIndexFromBinaryData(this->dataset, this->dataset_size);
		// this->index->write(); // write hercules tree +HNSW of leaf node
		this->num_leaf_node = this->index->first_node->num_leaf_node;
		cout<<"Number of leaf nodes: "<<this->num_leaf_node<<endl;
		cout<<"Number of internal nodes: "<<this->index->first_node->num_internal_node<<endl;

		this->leaves = this->index->getLeaves();

	}


	/* ******************************* 叶子节点id映射************************************** */
    void Hercules::generateLeafId2IdxMap(const char* output_path)
	{
		leafIndexToIdMap();
		writeLeafId2IdxFile(output_path);
	}

	//  建立叶子节点id到下标的映射，变成连续的数组
    void Hercules::leafIndexToIdMap(){
		for(int i=0;i<num_leaf_node;i++){
			Node* leaf=leaves[i];
			leafId2Idx[leaf->id]=i;
		}
	}

	// 将leafId2Idx写入文件
    void Hercules::writeLeafId2IdxFile(const char* output_path_) {
		char* output_path;
		char * leafId2Idx_path = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_txt) + strlen("leafId2Idx.txt"))));
		leafId2Idx_path = strcpy(leafId2Idx_path, this->index->index_setting->index_path_txt);
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
   ts_type* Hercules::generate_segment(Node* node, size_t* segments_dimension){

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
   void Hercules::generate_leafnode_file(){
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


    void Hercules::write_centroid_file(const char* method, leaf_centroid* centroids) {
	char path_buf[1024];
	snprintf(path_buf, sizeof(path_buf), "%sleaf_%s.txt", this->index->index_setting->index_path_txt, method);
	std::cout << "Path: " << path_buf << std::endl;


	std::ofstream ofs(path_buf);
	if (!ofs.is_open()) {
		std::cerr << "can not open file: " << path_buf << std::endl;
		exit(-1);
	}
	for (int i = 0; i < this->num_leaf_node; i++) {
		for (int j = 0; j < this->timeseries_size; j++) {
			ofs << std::fixed << centroids[i].ts_centroid[j];
			if (j != this->timeseries_size - 1) ofs << " ";
		}
		ofs << "\n";
	}
	std::cout << "Leaf centroids (" << method << ") written to " << path_buf << std::endl;
}

    void Hercules::generate_cluster_info_files(const char* output_path, int num_representatives) {
	if (this->num_leaf_node <= 0) {
		std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
		return;
	}

	// Buffers indexed by mapped leaf index (leafId2Idx ordering)
	std::vector<int> sizes(this->num_leaf_node, 0);
	std::vector<float> variances(this->num_leaf_node, 0.0f);
	std::vector<float> densities(this->num_leaf_node, 0.0f);
	std::vector<float> intra_distances(this->num_leaf_node, 0.0f);
	// representatives: flattened: for each leaf -> num_representatives * timeseries_size floats
	std::vector<float> representatives(this->num_leaf_node * num_representatives * this->timeseries_size, 0.0f);

	// Iterate over leaves and compute stats, but write into mapped index
	for (int i = 0; i < this->num_leaf_node; ++i) {
		Node *leaf = this->leaves[i];
		int rec_vecs_number = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;

		// Map to output index
		auto it = this->leafId2Idx.find(leaf->id);
		if (it == this->leafId2Idx.end()) {
			std::cerr << "Error: leaf->id " << leaf->id << " not found in leafId2Idx map!" << std::endl;
			return;
		}
		int out_idx = it->second;
		sizes[out_idx] = rec_vecs_number;

		// Load time series
		VectorWithIndex* rec = leaf->getTS(index);
		if (rec == nullptr) {
			std::cerr << "Error: Failed to get time series from leaf node." << std::endl;
			return;
		}

		// compute centroid for this leaf (use existing centroids_center if available)
		std::vector<float> centroid(this->timeseries_size, 0.0f);
		bool have_centroid = false;
		if (this->centroids_centroids != nullptr) {
			// centroids_centroids is ordered according to leafId2Idx; use out_idx
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_centroids[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		} else if (this->centroids_center != nullptr) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_center[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		}

		// If no precomputed centroid, compute mean
		if (!have_centroid) {
			for (int r = 0; r < rec_vecs_number; ++r) {
				for (int d = 0; d < this->timeseries_size; ++d) {
					centroid[d] += rec[r].ts[d];
				}
			}
			for (int d = 0; d < this->timeseries_size; ++d) centroid[d] /= std::max(1, rec_vecs_number);
		}

		// compute per-dimension variance and intra distances
		std::vector<double> var_per_dim(this->timeseries_size, 0.0);
		double intra_sum = 0.0;
		for (int r = 0; r < rec_vecs_number; ++r) {
			// distance to centroid
			double dist = 0.0;
			for (int d = 0; d < this->timeseries_size; ++d) {
				double diff = rec[r].ts[d] - centroid[d];
				var_per_dim[d] += diff * diff;
				dist += diff * diff;
			}
			intra_sum += sqrt(dist);
		}

		float mean_var = 0.0f; // Mean variance across dimensions
		if (rec_vecs_number > 1) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				var_per_dim[d] /= (rec_vecs_number - 1);
				mean_var += static_cast<float>(var_per_dim[d]);
			}
			mean_var /= this->timeseries_size;
		} else {
			mean_var = 0.0f;
		}
		variances[out_idx] = mean_var;

		if (rec_vecs_number > 0) {
			intra_distances[out_idx] = static_cast<float>(intra_sum / rec_vecs_number);
		} else {
			intra_distances[out_idx] = 0.0f;
		}

		// density heuristic: size / (variance^D + eps)
		double denom = pow(std::max(1e-6f, mean_var), this->timeseries_size);
		densities[out_idx] = static_cast<float>(rec_vecs_number / (denom + 1e-9));

		// representative vectors: choose closest to centroid up to num_representatives
		std::vector<std::pair<double,int>> dist_idx;
		dist_idx.reserve(rec_vecs_number);
		for (int r = 0; r < rec_vecs_number; ++r) {
			double dist = 0.0;
			for (int d = 0; d < this->timeseries_size; ++d) {
				double diff = rec[r].ts[d] - centroid[d];
				dist += diff * diff;
			}
			dist_idx.emplace_back(dist, r);
		}
		std::sort(dist_idx.begin(), dist_idx.end());

		int choose = std::min(num_representatives, rec_vecs_number);
		for (int rep = 0; rep < choose; ++rep) {
			int ridx = dist_idx[rep].second;
			int base = (out_idx * num_representatives + rep) * this->timeseries_size;
			for (int d = 0; d < this->timeseries_size; ++d) {
				representatives[base + d] = rec[ridx].ts[d];
			}
		}

		// free loaded rec
		for (int j = 0; j < rec_vecs_number; j++) {
			free(rec[j].ts);
			free(rec[j].ts_index);
		}
		free(rec);
	}

	// base dir derived from index_path_txt
	std::string base_dir = this->index->index_setting->index_path_txt;
	if (base_dir.back() != '/' && base_dir.back() != '\\') base_dir += "/";

	// sizes -> int32
	{
		std::vector<int32_t> sizes32(this->num_leaf_node);
		for (int i = 0; i < this->num_leaf_node; ++i) sizes32[i] = static_cast<int32_t>(sizes[i]);
		std::ofstream ofs(base_dir + "cluster_sizes.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_sizes.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(sizes32.data()), sizes32.size() * sizeof(int32_t));
	}

	// variances
	{
		std::ofstream ofs(base_dir + "cluster_variances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_variances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(variances.data()), variances.size() * sizeof(float));
	}

	// densities
	{
		std::ofstream ofs(base_dir + "cluster_densities.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_densities.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(densities.data()), densities.size() * sizeof(float));
	}

	// intra distances
	{
		std::ofstream ofs(base_dir + "intra_distances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open intra_distances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(intra_distances.data()), intra_distances.size() * sizeof(float));
	}

	// representatives
	{
		std::ofstream ofs(base_dir + "representatives.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open representatives.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(representatives.data()), representatives.size() * sizeof(float));
	}

}

    // 修正后的聚类信息生成函数（不包含representative_vectors）
    void Hercules::generate_cluster_info_files_corrected(const char* output_path) {
	if (this->num_leaf_node <= 0) {
		std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
		return;
	}

	// Buffers indexed by mapped leaf index (leafId2Idx ordering)
	std::vector<int> sizes(this->num_leaf_node, 0);  // 保持int类型，写入时使用int32格式
	std::vector<float> variances(this->num_leaf_node, 0.0f);
	std::vector<float> densities(this->num_leaf_node, 0.0f);
	std::vector<float> intra_distances(this->num_leaf_node, 0.0f);

	// Iterate over leaves and compute stats, but write into mapped index
	for (int i = 0; i < this->num_leaf_node; ++i) {
		Node *leaf = this->leaves[i];
		int rec_vecs_number = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;

		// Map to output index
		auto it = this->leafId2Idx.find(leaf->id);
		if (it == this->leafId2Idx.end()) {
			std::cerr << "Error: leaf->id " << leaf->id << " not found in leafId2Idx map!" << std::endl;
			return;
		}
		int out_idx = it->second;
		sizes[out_idx] = rec_vecs_number;  // 直接赋值整数

		// Load time series
		VectorWithIndex* rec = leaf->getTS(index);
		if (rec == nullptr) {
			std::cerr << "Error: Failed to get time series from leaf node." << std::endl;
			return;
		}

		// compute centroid for this leaf (use existing centroids_center if available)
		std::vector<float> centroid(this->timeseries_size, 0.0f);
		bool have_centroid = false;
		if (this->centroids_centroids != nullptr) {
			// centroids_centroids is ordered according to leafId2Idx; use out_idx
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_centroids[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		} else if (this->centroids_center != nullptr) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_center[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		}

		// If no precomputed centroid, compute mean
		if (!have_centroid) {
			for (int r = 0; r < rec_vecs_number; ++r) {
				for (int d = 0; d < this->timeseries_size; ++d) {
					centroid[d] += rec[r].ts[d];
				}
			}
			for (int d = 0; d < this->timeseries_size; ++d) centroid[d] /= std::max(1, rec_vecs_number);
		}

		// compute per-dimension variance and intra distances
		std::vector<double> var_per_dim(this->timeseries_size, 0.0);
		double intra_sum = 0.0;
		for (int r = 0; r < rec_vecs_number; ++r) {
			// distance to centroid
			double dist = 0.0;
			for (int d = 0; d < this->timeseries_size; ++d) {
				double diff = rec[r].ts[d] - centroid[d];
				var_per_dim[d] += diff * diff;
				dist += diff * diff;
			}
			intra_sum += sqrt(dist);
		}

		float mean_var = 0.0f; // Mean variance across dimensions
		if (rec_vecs_number > 1) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				var_per_dim[d] /= (rec_vecs_number - 1);
				mean_var += static_cast<float>(var_per_dim[d]);
			}
			mean_var /= this->timeseries_size;
		} else {
			mean_var = 0.0f;
		}
		variances[out_idx] = mean_var;

		if (rec_vecs_number > 0) {
			intra_distances[out_idx] = static_cast<float>(intra_sum / rec_vecs_number);
		} else {
			intra_distances[out_idx] = 0.0f;
		}

		// density heuristic: size / (variance^D + eps)
		double denom = pow(std::max(1e-6f, mean_var), this->timeseries_size);
		densities[out_idx] = static_cast<float>(rec_vecs_number / (denom + 1e-9));

		// free loaded rec
		for (int j = 0; j < rec_vecs_number; j++) {
			free(rec[j].ts);
			free(rec[j].ts_index);
		}
		free(rec);
	}

	// base dir derived from index_path_txt
	std::string base_dir = this->index->index_setting->index_path_txt;
	if (base_dir.back() != '/' && base_dir.back() != '\\') base_dir += "/";

	// cluster_sizes -> int32 (保持与原版本一致)
	{
		std::ofstream ofs(base_dir + "cluster_sizes.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_sizes.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(sizes.data()), sizes.size() * sizeof(int));
		std::cout << "cluster_sizes.bin written: " << sizes.size() << " ints" << std::endl;
	}

	// variances
	{
		std::ofstream ofs(base_dir + "cluster_variances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_variances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(variances.data()), variances.size() * sizeof(float));
		std::cout << "cluster_variances.bin written: " << variances.size() << " floats" << std::endl;
	}

	// densities
	{
		std::ofstream ofs(base_dir + "cluster_densities.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_densities.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(densities.data()), densities.size() * sizeof(float));
		std::cout << "cluster_densities.bin written: " << densities.size() << " floats" << std::endl;
	}

	// intra distances
	{
		std::ofstream ofs(base_dir + "intra_distances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open intra_distances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(intra_distances.data()), intra_distances.size() * sizeof(float));
		std::cout << "intra_distances.bin written: " << intra_distances.size() << " floats" << std::endl;
	}

	std::cout << "All cluster info files generated successfully!" << std::endl;
}


    void Hercules::generate_leaf_centroids(const char* output_path) {
	if (this->num_leaf_node <= 0) {
		std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
		return;
	}

	// 为两种方法分别分配临时质心数组
	leaf_centroid* centroids_centroid = (leaf_centroid*)malloc_index(sizeof(leaf_centroid) * this->num_leaf_node);
	leaf_centroid* centroids_center   = (leaf_centroid*)malloc_index(sizeof(leaf_centroid) * this->num_leaf_node);

	// cout<<" 临时质心数组生成"<<endl;
	// 临时质心数组生成
	for (int i = 0; i < this->num_leaf_node; i++) {
		Node* leaf = this->leaves[i];
		int rec_vecs_number = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;

		// 分配空间
		centroids_centroid[i].ts_centroid = (ts_type*)calloc(this->timeseries_size, sizeof(ts_type));
		centroids_center[i].ts_centroid   = (ts_type*)calloc(this->timeseries_size, sizeof(ts_type));
		centroids_centroid[i].leaf_centroid_index = (file_position_type*)malloc_index(sizeof(file_position_type));
		centroids_center[i].leaf_centroid_index   = (file_position_type*)malloc_index(sizeof(file_position_type));
		*(centroids_centroid[i].leaf_centroid_index) = leaf->id;
		*(centroids_center[i].leaf_centroid_index) = leaf->id;

		// 获取向量
		VectorWithIndex* rec = leaf->getTS(index);
		if (rec == nullptr) {
			std::cerr << "Error: Failed to get time series from leaf node." << std::endl;
			exit(-1);
		}

		/* cout<<"计算Center(最小距离点）"<<endl;
		// --- 计算 "Center"（最小距离点） ---
		file_position_type center_index = 0;
		double min_distance = std::numeric_limits<double>::infinity();
		for (int j = 0; j < rec_vecs_number; j++) {
			double total_distance = 0.0;
			for (int k = 0; k < rec_vecs_number; k++) {
				if (j != k)
					total_distance += euclideanDistance(rec[j].ts, rec[k].ts, this->timeseries_size);
			}
			if (total_distance < min_distance) {
				min_distance = total_distance;
				center_index = j;
			}
		}
		for (int j = 0; j < this->timeseries_size; j++)
			centroids_center[i].ts_centroid[j] = rec[center_index].ts[j]; */

        // 为当前聚类分配中位数向量空间
        std::vector<float> median_center(this->timeseries_size);
        // 对每个维度，计算该维度的中位数
        for (int dim = 0; dim < this->timeseries_size; dim++) {
        	std::vector<float> dim_values(rec_vecs_number);
        	for (int j = 0; j < rec_vecs_number; j++) {
        		dim_values[j] = rec[j].ts[dim];
        	}
        	std::nth_element(dim_values.begin(), dim_values.begin() + rec_vecs_number / 2, dim_values.end());
        	if (rec_vecs_number % 2 == 1) {
        		median_center[dim] = dim_values[rec_vecs_number / 2];
        	} else {
        		std::nth_element(dim_values.begin(), dim_values.begin() + rec_vecs_number / 2 - 1, dim_values.end());
        		float a = dim_values[rec_vecs_number / 2];
        		float b = dim_values[rec_vecs_number / 2 - 1];
        		median_center[dim] = (a + b) / 2.0f;
        	}
        }
        
        // 将中位数向量拷贝到当前质心数组
        for (int j = 0; j < this->timeseries_size; j++) {
        	centroids_center[i].ts_centroid[j] = median_center[j];
        }


		// --- 计算 "Centroid"（平均向量） ---
		for (int j = 0; j < rec_vecs_number; j++) {
			for (int k = 0; k < this->timeseries_size; k++) {
				centroids_centroid[i].ts_centroid[k] += rec[j].ts[k];
			}
		}
		for (int k = 0; k < this->timeseries_size; k++) {
			centroids_centroid[i].ts_centroid[k] /= rec_vecs_number;
		}

		// 清理
		for (int j = 0; j < rec_vecs_number; j++) {
			free(rec[j].ts);
			free(rec[j].ts_index);
		}
		free(rec);
	}


	cout<<"将质心位置映射和label一致"<<endl;
	// 将质心位置映射和label一致
	// 分配内存
	this->centroids_centroids = static_cast<leaf_centroid *>(malloc_index(sizeof(leaf_centroid) * this->num_leaf_node));
	this->centroids_center = static_cast<leaf_centroid *>(malloc_index(sizeof(leaf_centroid) * this->num_leaf_node));
	for(int i = 0; i < this->num_leaf_node; i++) {
		this->centroids_centroids[i].ts_centroid = static_cast<ts_type *>(malloc_index(sizeof(ts_type) * this->timeseries_size));
		this->centroids_centroids[i].leaf_centroid_index = static_cast<file_position_type *>(malloc_index(sizeof(file_position_type)));
		*(this->centroids_centroids[i].leaf_centroid_index) = 0; // 初始化为0，后续会被覆盖
		this->centroids_center[i].ts_centroid = static_cast<ts_type *>(malloc_index(sizeof(ts_type) * this->timeseries_size));
		this->centroids_center[i].leaf_centroid_index = static_cast<file_position_type *>(malloc_index(sizeof(file_position_type)));
		*(this->centroids_center[i].leaf_centroid_index) = 0; // 初始化为0，后续会被覆盖
	}
	// 处理centriods
	for(int i = 0; i < this->num_leaf_node; i++) {
	    int idx_leaf=*(centroids_centroid[i].leaf_centroid_index);
	    int idx = this->leafId2Idx[idx_leaf];
	    for(int j=0;j<this->timeseries_size;j++){
	    	this->centroids_centroids[idx].ts_centroid[j] = centroids_centroid[i].ts_centroid[j];
	    }
	    *(this->centroids_centroids[idx].leaf_centroid_index) = *(centroids_centroid[i].leaf_centroid_index);
	}
	// 处理center
	for(int i = 0; i < this->num_leaf_node; i++) {
	    int idx_leaf=*(centroids_center[i].leaf_centroid_index);
	    int idx = this->leafId2Idx[idx_leaf];
	    for(int j=0;j<this->timeseries_size;j++){
	    	this->centroids_center[idx].ts_centroid[j] = centroids_center[i].ts_centroid[j];
	    }
	    *(this->centroids_center[idx].leaf_centroid_index) = *(centroids_center[i].leaf_centroid_index);
	}


	// 写出两种质心方式
	write_centroid_file("centroid", this->centroids_centroids);
	write_centroid_file("center", this->centroids_center); 

	// 释放内存
	for (int i = 0; i < this->num_leaf_node; i++) {
		free(centroids_centroid[i].ts_centroid);
		free(centroids_center[i].ts_centroid);
		free(centroids_centroid[i].leaf_centroid_index);
		free(centroids_center[i].leaf_centroid_index);
	}
	free(centroids_centroid);
	free(centroids_center);
}


    // 写出叶子包含的向量数目（顺序与质心一致）
    void Hercules::write_leaf_sizes_file(const char* output_path_) {
    	if (this->num_leaf_node <= 0) {
    		std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
    		return;
    	}

    	// 先按 leafId2Idx 的顺序放置大小
    	int* leaf_sizes = static_cast<int*>(calloc(this->num_leaf_node, sizeof(int)));
    	for (int i = 0; i < this->num_leaf_node; i++) {
    		Node *leaf = this->leaves[i];
    		int count = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;
    		int idx = this->leafId2Idx[leaf->id];
    		leaf_sizes[idx] = count;
    	}

    	char path_buf[1024];
    	if (output_path_ == nullptr || strlen(output_path_) == 0) {
    		snprintf(path_buf, sizeof(path_buf), "%sleaf_size.txt", this->index->index_setting->index_path_txt);
    	} else {
    		strncpy(path_buf, output_path_, sizeof(path_buf));
    		path_buf[sizeof(path_buf)-1] = '\0';
    	}

    	std::ofstream ofs(path_buf);
    	if (!ofs.is_open()) {
    		std::cerr << "can not open file: " << path_buf << std::endl;
    		exit(-1);
    	}
    	for (int i = 0; i < this->num_leaf_node; i++) {
    		ofs << leaf_sizes[i] << "\n";
    	}
    	ofs.close();
    	free(leaf_sizes);
    	std::cout << "Leaf sizes written to " << path_buf << std::endl;
    }

    /* ******************************* 建立时间序列id到叶子节点的映射，便于生成label************************************** */
 	void Hercules::generate_ts_leaf_map_file(const char* output_path){
		fillTsLeafMap();
		writeTsLeafMapFile(output_path);
	}


    // 填充哈希表
	void Hercules::fillTsLeafMap()
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

	void Hercules::writeTsLeafMapFile(const char* output_path_)
	{

		char* output_path;
		char * ts_leaf_map = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_txt) + strlen("/ts_leaf_map.txt"))));
		ts_leaf_map = strcpy( ts_leaf_map , this->index->index_setting->index_path_txt);
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
			out<<"ts_key: "<<ts_key;
			out<<" leaf->id: "<<leaf->id;
			out<< endl;
		}
		std::cout << "ts_leaf_map.txt has been written successfully." << std::endl;
	}

	/* ******************************* top-k groundtruth----> leaf-id *************************** */
	void Hercules::topk2LeafId(const char* output_path)
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
					cerr<<"ts_key="<<ts_key<<" is not between[0,"<<this->dataset_size<<"]"<<endl;
					exit(-1);
				}

				auto it = this->ts_leaf_map.find(ts_key);
				Node *leaf=nullptr;
                if (it != this->ts_leaf_map.end()) {
                    leaf = it->second;
					if (leaf == nullptr) {
                       cerr << "Error: Node* for key " << ts_key << " is nullptr!" << std::endl;
                       exit(-1);
                   }
                } else{
				    cerr << "Error: Key not found: " << ts_key << std::endl;
					exit(-1);
				}
                knn_groundtruth[i][j]= leaf->id; // 将叶子节点的id存入knn_groundtruth[i][j]

			}
		}

	    // 输出knn_groundtruth的内容到文件中
		char * knn_groundtruth_path = static_cast<char *>(malloc(sizeof(char) * (strlen(this->index->index_setting->index_path_txt) + strlen("knn_groundtruth.txt")+1)));
		knn_groundtruth_path = strcpy(knn_groundtruth_path, this->index->index_setting->index_path_txt);
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

	void Hercules::leafContainsTopK(int selected_k)
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


	/* ******************************* 生成标签 knn_distribution************************************** */
	void Hercules::generate_label(unsigned int selected_k, const char* output_path){
		calcKNNinLeaves(selected_k);
		writeKNNDistributionsToFile(selected_k,output_path);
	}
	
	// 生成label：每个query的top-k在叶子中的分布
	void Hercules::calcKNNinLeaves(unsigned int selected_k)
	{
		this->knn_distributions = new file_position_type *[this->groundtruth_dataset_size];

		selected_k=std::min<unsigned int>(selected_k,this->groundtruth_top_k);

		for (int i = 0; i < this->groundtruth_dataset_size; i++)
		{
			this->knn_distributions[i] = new file_position_type[this->num_leaf_node];
			memset(this->knn_distributions[i], 0, this->num_leaf_node * sizeof(file_position_type));
			for (int j = 0; j < selected_k; j++)
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

			// 检查knn_distribution[i]所有元素的和是否为selected_k
			file_position_type sum = 0;
			for (int k = 0; k < this->num_leaf_node; k++) {
				sum += knn_distributions[i][k];
			}
			if (sum != selected_k) {
				cout<<"knn_distribution["<<i<<"] sum" << sum<<"!= selected_k"<<endl;
				exit(-1);
			}

			/* // 输出knn_distributions[i]的内容
			cout<<"knn_distributions["<<i<<"]: ";
			for(int k=0;k<this->num_leaf_node;k++){
				cout<<knn_distributions[i][k]<<" ";
			}
			cout<<endl; */
		}
	}

    void Hercules::TrainWeight(){
		/* learn_dataset训练权重 */

		this->fillTsLeafMap();
		this->getCandidateLeafNode(100);

		// this->index->first_node->num_leaf_node = 0;
		// this->index->first_node->num_internal_node = 0;
		// this->index = Index::Read(this->index_path, 1); 

        this->queryengine = new QueryEngine(this->query_dataset, this->query_dataset_size, 
                                            this->groundtruth_dataset , this->groundtruth_top_k, this->groundtruth_dataset_size, 
                                            this->learn_dataset, this->learn_dataset_size, this->learn_groundtruth_dataset,
                                            this->dataset, 
											this->index, this->efSearch, this->nprobes, this->parallel, 
											this->nworker, this->flatt, this->k, this->ep, 
											this->model_file, this->zero_edge_pass_ratio);
		this->searchCandidateLeafNode();

		this->queryengine->queryBinaryFile(this->k, this->mode, this->search_withWeight, this->thres_probability, this->μ, this->T);

		cout << "[Querying Time] "<< this->index->time_stats->querying_time <<"(sec)"<<endl;  
        cout << "[QPS] "<< query_dataset_size*1.0/index->time_stats->querying_time <<endl;  
		double averageRecall = this->queryengine->calculateAverageRecall();
		cout << "[Average Recall] "<< averageRecall << endl;
		this->index->write(); // write hercules tree +HNSW of leaf node
	}

	
	void Hercules::getCandidateLeafNode(unsigned int selected_k){
		/**
		 * 如果一个查询向量的top-k在某个叶子中，那么这个叶子就是候选叶子，记住该叶子节点。如果一个查询向量的top-k不在某个叶子中，那么这个叶子就是非候选叶子，无需记录在candidate_leaf_node中
		 * 每个查询向量的候选叶子节点保存到candidate_leaf_node中
		 * candidate_leaf_node[i][j]表示第i个查询向量的第j个候选叶子节点的id
		 * 每个candidate_leaf_node[i]包含的候选叶子的数目不一样
		 */
        this->candidate_leaf_node.clear();                     // 可选：先清空旧内容
        this->candidate_leaf_node.resize(this->learn_dataset_size);
		selected_k=std::min<unsigned int>(selected_k,this->groundtruth_top_k);
		for (size_t i = 0; i < this->learn_dataset_size; i++)
		{
			for (size_t j = 0; j < selected_k; j++)
			{
				int ts_key = this->learn_groundtruth_list[i][j];
				if (ts_key >= this->dataset_size || ts_key < 0) {
					cout << "ts_key=" << ts_key << " is not between[0," << this->dataset_size << "]" << endl;
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
				}
				else {
					std::cerr << "Error: Key not found: " << ts_key << std::endl;
					exit(-1);
				}
				// 将该叶子节点id插入到candidate_leaf_node[i]中
				this->candidate_leaf_node[i].insert(leaf);
			}
		}
	}

	void Hercules::leafNodeID2Node(){
		/*
		* 将叶子节点ID映射到 节点
		*/
		for (int i = 0; i < this->num_leaf_node; i++)
		{
			Node* leaf = this->leaves[i];
			int leaf_id = leaf->id;
			this->leafNode2GraphMap[leaf_id] = leaf;
		}
	}

	void Hercules::searchCandidateLeafNode(){
		/*
		* 对于每个查询向量，搜索所有candidate leaf node
		* 对该候选叶子节点进行边权训练
		*/
		this->queryengine->TrainWeightByLearnDataset(this->queryengine->ep, this->queryengine->k , this->candidate_leaf_node);

	}

	void Hercules::writeKNNDistributionsToFile(unsigned int selected_k,const char* output_path_) {
		char* output_path;
	
		// 为文件路径分配足够的空间，多预留 selected_k 数字的空间
		size_t buffer_size = strlen(this->index->index_setting->index_path_txt) 
						   + strlen("knn_distributions_k_.txt") 
						   + 20; // 预留数字长度
		char * knn_distribution_path = static_cast<char *>(malloc(sizeof(char) * buffer_size));
	
		// 拼接路径和文件名
		sprintf(knn_distribution_path, "%sknn_distributions_k%d.txt", 
				this->index->index_setting->index_path_txt, selected_k);
	
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
	
		// 释放内存
		if (output_path == knn_distribution_path) {
			free(knn_distribution_path);
		}
	}

	
	void Hercules::generateAllFiles(){
		
		// 生成 base向量id到 叶子节点id的映射文件
       this->generate_ts_leaf_map_file();

	   // top-k groundtruth----> leaf-id
       this->topk2LeafId();
       //  建立叶子节点id到下标的映射，变成连续的数组
       this->generateLeafId2IdxMap();
       //    hercules->leafContainsTopK(20);
       //    hercules->leafContainsTopK(10);
       //    hercules->leafContainsTopK(30);
       //    hercules->leafContainsTopK(50);
       //    hercules->leafContainsTopK(100);

        // 生成叶子节点文件以及质心文件
       this->generate_leafnode_file();
       this->generate_leaf_centroids();
	   this->write_leaf_sizes_file();
	  // 生成每个叶子的聚类信息（与质心顺序一致）
	  //  generate_cluster_info_files();
		// generate_cluster_info_files_corrected();
       // 生成标签
       this->generate_label(1);
	   this->generate_label(10);
	   this->generate_label(20);
	   this->generate_label(50);
	   this->generate_label(100);

	}

	// 析构函数
	Hercules::~Hercules() {
		if (this->index) {
			delete this->index;
		}
		if (this->ts_list) {
			for (int i = 0; i < this->dataset_size; i++) {
				free(this->ts_list[i].ts);
				delete this->ts_list[i].ts_index;  // 修复：使用 delete 而不是 free
			}
			delete[] this->ts_list;  // 修复：使用 delete[] 而不是 free
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
		if (this->centroids_centroids) {
			for (int i = 0; i < this->num_leaf_node; i++) {
				free(this->centroids_centroids[i].ts_centroid);
				free(this->centroids_centroids[i].leaf_centroid_index);
			}
			free(this->centroids_centroids);
		}
		if (this->centroids_center) {
			for (int i = 0; i < this->num_leaf_node; i++) {
				free(this->centroids_center[i].ts_centroid);
				free(this->centroids_center[i].leaf_centroid_index);
			}
			free(this->centroids_center);
		}
		if (this->knn_distributions) {
			for (int i = 0; i < this->groundtruth_dataset_size; i++) {
				delete[] this->knn_distributions[i];  // 修复：使用 delete[] 而不是 free
			}
			delete[] this->knn_distributions;
		}
		if(this->knn_groundtruth) {
			for (int i = 0; i < this->groundtruth_dataset_size; i++) {
				delete[] this->knn_groundtruth[i];  // 修复：使用 delete[] 而不是 free
			}
			delete[] this->knn_groundtruth;
		}
		if(this->queryengine) {
			delete this->queryengine;
		}
	}
