#include "BufferManager.h"


BufferManager::BufferManager(Setting *index_setting) {

    this -> max_buffered_size = 1000*1000*100;

    this->current_count = 0;

    this->file_map = NULL;                                                       // 双端链表的头
    this->file_map_tail = NULL;                                                  // 双端链表的尾
    this->file_map_size = 0;

    // unsigned long num_bytes = index_setting->buffered_memory_size * 1024 * 1024; 
    unsigned long num_bytes = index_setting->buffered_memory_size * 1024 ; 

    this->max_buffered_size = (long)( num_bytes / sizeof(ts_type));
    this->batch_remove_size = this->max_buffered_size/2;                        // 将缓冲区域一分为二，一边读入，一边写出

    int max_leaf_size = index_setting->max_leaf_size;
    unsigned long leaf_size = index_setting->max_leaf_size;
    unsigned long ts_size = sizeof(ts_type) * index_setting->timeseries_size;   // 每个向量占据的字节数

    unsigned long num_leaf_buffers = 2*((long)( num_bytes / (ts_size*leaf_size) ));


    unsigned long size_leaf_buffer = sizeof(struct hercules_file_buffer) + (sizeof(ts_type *) * max_leaf_size);

    long long mem_array_size = (long long) ((num_bytes -  size_leaf_buffer * num_leaf_buffers) / ts_size);

    this->mem_array = static_cast<char *>(calloc(mem_array_size, ts_size));

    if(this->mem_array == NULL){
        std::cerr << "Error while allocating space for BufferManager->Mem_Array, we can not allocate space of size "
                  <<index_setting->buffered_memory_size<<std::endl;
        std::cerr << "ts_size_bytes : " << ts_size
                  << "| mem_array_size : "  << mem_array_size 
                  << "| mem_array : "<< this->mem_array 
                  << std::endl;
        exit(FAILURE);
    }
    
    fprintf(stderr,">>>>  MEM_ARRAY first address : %p \n",this->mem_array);
    this->current_record_index = 0;
    this->max_record_index = mem_array_size;
    this->current_record = this->mem_array;
    this->toString();
}


void BufferManager::toString(){
    std::cerr << "[Buffer Manager] Current account " << this->current_count
    << " - current_record_index " << this->current_record_index
    << " - max_buffered_size " << this->max_buffered_size
    << " - file_map_size " << this->file_map_size
    << " - current_record in @ " ;
    fprintf(stderr,"%s \n",this->current_record);
}



