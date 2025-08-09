unordered_map<string, Node *> ts_leaf_map; // Using string as key for vector<ts_type> serialization
this->ts_leaf_map.reserve(this->dataset_size*2);
this->ts_leaf_map.max_load_factor(0.5); // 默认是 1.0，调整为更低的值（例如 0.7）可以减少冲突


// 填充哈希表
	void fillTsLeafMap()
	{
		// this->ts_leaf_map.reserve(this->dataset_size/2);
		for (int i = 0; i < this->num_leaf_node; i++)
		{
			Node *leaf = this->leaves[i];
			ts_type **rec = leaf->getTS(index);
			int rec_vecs_number=leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;
			for (int j = 0; j < rec_vecs_number; j++)
			{
				ts_type *ts=rec[j];
				std::ostringstream oss;
                for (int k = 0; k < this->timeseries_size; ++k)
                {
					if(k>0) oss<<" ";
                    oss << ts[k];
                }
                string ts_key = oss.str();
                // this->ts_leaf_map[ts_key] = leaf;
				auto result = this->ts_leaf_map.insert({ts_key, leaf});
                if (!result.second) {
                    // 插入失败，说明键已经存在
                    std::cerr << "Key already exists: " << ts_key << std::endl;
                }

			}
			for(int i=0;i<rec_vecs_number;i++){
				free(rec[i]);
			}
			free(rec);
		}
	}


    // 生成label：每个query的top-100在叶子中的分布
	void calcKNNinLeaves()
	{
		knn_distributions = new float *[this->query_dataset_size];
		for(int i=0;i<groundtruth_dataset_size;i++){
			knn_distributions[i] = new float[this->num_leaf_node];
			memset(knn_distributions[i], 0, this->num_leaf_node * sizeof(float));
		}
		
		for (int i = 0; i < groundtruth_dataset_size; i++)
		{
			for (int j = 0; j < 100; j++)
			{
				int id=this->groundtruth_list[i][j];
				if(id>=this->dataset_size){
					cout<<"id>=this->dataset_size"<<endl;
					exit(-1);
				}


				ts_type *ts = this->ts_list[id];
				std::ostringstream oss;
                for (int k = 0; k < this->timeseries_size; ++k)
                {
					if(k>0) oss<<" ";
                    oss << ts[k];
                }
                string ts_key = oss.str();

				// cout<<ts_key<<endl;
				auto it = this->ts_leaf_map.find(ts_key);
				Node *leaf=nullptr;
                if (it != this->ts_leaf_map.end()) {
                    leaf = it->second;
                } else{
				    std::cerr << "Error: Key not found: " << ts_key << std::endl;
                    // 处理未找到的情况，比如返回默认值，或者继续处理
				}
				// 查看是否有hash冲突的现象
				// std::cout << "Hash value for key: " << std::hash<std::string>{}(ts_key) << std::endl;

				if(leaf==nullptr){
					cout<<ts_key<<endl;
					cout<<id<<endl;
					exit(-1);
				}


				// Node *leaf = this->ts_leaf_map[ts_key];
				++knn_distributions[i][leaf->id];
			}
		}

	}

