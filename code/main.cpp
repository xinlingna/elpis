#include <iostream>
#include <getopt.h>
#include "Index.h"
#include "Hercules.h"
#include "QueryEngine.h"
#include  <stdlib.h>
#include <random>


using namespace std;
int main(int argc, char **argv) {

    /* hercules tree args */
    static char *dataset = "nodataset";
    // static char *queries = "noquery";
    static char * index_path = "index/";
    static unsigned int dataset_size = 1000;
    // static unsigned int queries_size = 5;
    static unsigned int time_series_size = 256;
    static unsigned int init_segments = 1;
    static unsigned int leaf_size = 100;
    static double buffered_memory_size = 64.2;
    static int use_ascii_input = 0;
    int ef = 10;
    static int mode = 1;

    static unsigned int k = 1;
    static unsigned int nprobes = 10;
    bool parallel = true;
    static unsigned int nworker = 0;
    
    /* HNSW args */
    static int efConstruction = 500;
    static int M = 4 ;
    static bool flatt = 0;

    static char *query_dataset = nullptr;
    static unsigned int query_dataset_size = 0;
    static char *groundtruth_dataset = nullptr;
    static unsigned int groundtruth_dataset_size = 0;
    static unsigned int groundtruth_top_k = 0;

    /* HNSW weight para */
    unsigned int ep=20;
    static char* learn_dataset=nullptr;
    static char* learn_groundtruth_dataset=nullptr;
    unsigned int learn_dataset_size= 0;
    float thres_probability=0.3;
    float μ=0.0;
    float T=1.0;
    float zero_edge_pass_ratio = 0.0f; // ρ

    /* model para */
    const char* model_file=nullptr;

    while (1) {
        static struct option long_options[] = {
                {"ascii-input",      required_argument, 0, 'a'},
                {"buffer-size",      required_argument, 0, 'b'},
                {"epsilon",          required_argument, 0, 'c'},
                {"kb",               required_argument, 0, 'mh'},
                {"flatt",            required_argument, 0, 'ft'},
                {"Lb",               required_argument, 0, 'efc'},
                {"L",                required_argument, 0, 'ef'},
                {"dataset",          required_argument, 0, 'd'},
                {"dataset-hists",    required_argument, 0, 'n'},
                {"delta",            required_argument, 0, 'e'},
                {"queries-size",     required_argument, 0, 'f'},
                {"track-bsf",        required_argument, 0, 'g'},
                {"track-pruning",    required_argument, 0, 'i'},
                {"all-mindists",     required_argument, 0, 'j'},
                {"max-policy",       required_argument, 0, 'm'},
                {"queries",          required_argument, 0, 'q'},
                {"index-path",       required_argument, 0, 'p'},
                {"dataset-size",     required_argument, 0, 'z'},
                {"k",                required_argument, 0, 'k'},
                {"mode",             required_argument, 0, 'x'},
                {"in-memory",        required_argument, 0, 'im'},
                {"minimum-distance", required_argument, 0, 's'},
                {"timeseries-size",  required_argument, 0, 't'},
                {"leaf-size",        required_argument, 0, 'l'},
                {"nprobes",          required_argument, 0, 'o'},
                {"parallel",         required_argument, 0, 'pr'},
                {"nworker",          required_argument, 0, 'nw'},

                {"incremental",              no_argument,       0, 'h'},
                {"index-path-hercules",      required_argument, 0, 'pd'},
                {"index-path-hnsw",          required_argument, 0, 'ph'},
                {"help",                     no_argument,       0, '?'},

                {"query_dataset",            required_argument, 0, 2001},
                {"query_dataset_size",       required_argument, 0, 2002},
                {"groundtruth_dataset",      required_argument, 0, 2003},
                {"groundtruth_dataset_size", required_argument, 0, 2004},
                {"groundtruth_top_k",        required_argument, 0, 2005},

                {"ep",                         required_argument, 0, 2006},
                {"learn_dataset",              required_argument, 0, 2007},
                {"learn_dataset_size",         required_argument, 0, 2008},
                {"learn_groundtruth_dataset",  required_argument, 0, 2009},
                {"model_file",                 required_argument, 0, 2010},
                {"thres_probability",          required_argument, 0, 2011},
                {"μ",                          required_argument, 0, 2012},
                {"T",                          required_argument, 0, 2013}, 
                {"zero_edge_pass_ratio",      required_argument, 0, 2014},
        };

        // getopt_long stores the option index here.
        int option_index = 0;

        int c = getopt_long(argc, argv, "",
                            long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {

            case 'pr':
                parallel = atoi(optarg);
                break;
            case 'nw':
                nworker = atoi(optarg);
                break;
            case 'efc':
                efConstruction = atoi(optarg);
                if(efConstruction<1){
                    cerr <<"Please give a correct efconstruction >1"<<endl;
                    exit(-1);
                }
                break;
            case 'ef':
                ef = atoi(optarg);
                break;

            case 'mh':
                M = atoi(optarg);                  //////////////////////
                if(M<1){
                    cerr <<"Please give a correct M >1"<<endl;
                    exit(-1);
                }
                break;
            case 'ft':
                flatt = atoi(optarg);
                break;

            // case 'q':
            //     queries = optarg;
            //     break;


            case 'b':
                buffered_memory_size = atof(optarg);
                break;

/*             case 'f':
                queries_size = atoi(optarg);
                if (queries_size < 1) {
                    fprintf(stderr, "Please change the queries size to be greater than 0.\n");
                    exit(-1);
                }
                break; */

            case 'k':
                k = atoi(optarg);
                if (k < 1) {
                    fprintf(stderr, "Please change the k to be greater than 0.\n");
                    exit(-1);
                }
                break;


            case 'd':
                dataset = optarg;
                break;
            case 'p':
                index_path = optarg;
                break;

            case 'x':
                mode = atoi(optarg);
                break;
            case 'z':
                dataset_size = atoi(optarg);
                if (dataset_size < 1) {
                    fprintf(stderr, "Please change the dataset size to be greater than 0.\n");
                    exit(-1);
                }
                break;

            case 't':
                time_series_size = atoi(optarg);
                break;

            case 'o':
                nprobes = atoi(optarg);
                break;

            case 'l':
                leaf_size = atoi(optarg);
                if (leaf_size <= 1) {
                    fprintf(stderr, "Please change the leaf size to be greater than 1.\n");
                    exit(-1);
                }
                break;

            case 2001:
                query_dataset = optarg;
                break;
            case 2002:
                query_dataset_size = atoi(optarg);
                break;
            case 2003:
                groundtruth_dataset = optarg;
                break;
            case 2004:
                groundtruth_dataset_size = atoi(optarg);
                break;
            case 2005:
                groundtruth_top_k = atoi(optarg);
                break;
            case 2006:
                ep = atoi(optarg);
                break;
            case 2007:
                learn_dataset = optarg;
                break;
            case 2008:
                learn_dataset_size = atoi(optarg);
                break;
            case 2009:
                learn_groundtruth_dataset = optarg;
                break;
            case 2011:
                thres_probability = atof(optarg);
                break;
            case 2012:
                μ = atof(optarg);
                break;
            case 2013:  
                T = atof(optarg);
                break;
            case 2014:
                zero_edge_pass_ratio = atof(optarg);
                if (zero_edge_pass_ratio < 0.0f) zero_edge_pass_ratio = 0.0f;
                if (zero_edge_pass_ratio > 1.0f) zero_edge_pass_ratio = 1.0f;
                break;

            case '?':

                printf("Usage:\n"
                       "\t--Queries and Datasets should be single precision floating points\n"
                       "\t--They can be in either binary or ASCII format.\n"
                       "\t--However, performance is faster with binary files.\n\n"
                       "\tOptions:\n"
                       "\t--dataset XX \t\t\tThe path to the dataset file.\n"
                       "\t--queries XX \t\t\tThe path to the queries file.\n"
                       "\t--dataset-size XX \t\tThe number of time series to load.\n"
                       "\t--queries-size XX \t\tThe number of queries to run.\n"
                       "\t--mode XX \t\t\tMode of operation: 0=index, 1=query, 2=index & query, 3=calc_tlb.\n"
                       "\t--index-path XX \t\tThe path of the output folder.\n"
                       "\t--buffer-size XX \t\tThe size of the buffer memory in MB.\n"
                       "\t--timeseries-size XX \tThe size of each time series.\n"
                       "\t--ascii-input X \t\t0 for ASCII files, 1 for binary files.\n"
                       "\t--leaf-size XX \t\tThe maximum size of each leaf node.\n"
                       "\t--M XX \t\t\tParameter defining the maximum number of outgoing connections in leaf graphs.\n"
                       "\t--efconstruction XX \t\tControls speed/accuracy trade-off during leaf index construction.\n"
                       "\t--parallel XX \t\tSet to 1 for parallel querying.\n"
                       "\t--nworker XX \t\t\tNumber of workers for parallel querying. Default is the number of cores - 1.\n"
                       "\t--help \t\t\tShow this help message.\n\n"
                       "\t**********************EXAMPLES**********************\n\n"
                       "\t*********************INDEX MODE*********************\n"
                       "\t--bin/hercules --dataset XX --dataset-size XX \n"
                       "\t--          --index-path XX --timeseries-size XX --mode 0\n\n"
                       "\t*********************QUERY MODE*********************\n"
                       "\t--bin/hercules --queries XX --queries-size XX \n"
                       "\t--           --index-path XX --mode 1\n\n"
                       "\t*****************INDEX AND QUERY MODE***************\n"
                       "\t--bin/hercules --dataset XX --dataset-size XX \n"
                       "\t--          --timeseries-size XX --index-path XX\n"
                       "\t--           --queries XX --queries-size XX --mode 2\n\n"
                       "\t****************************************************\n");
 
                return 0;
                break;
            case 'a':
                use_ascii_input = atoi(optarg);
                break;

            default:
                exit(-1);
                break;
        }
    }


    ///CREATE INDEX
    if(mode == 0){  /* 存在内存泄漏的问题，需要修改 */
    Index * index = Index::initIndex(index_path, time_series_size, buffered_memory_size*1024, init_segments, 
        leaf_size, efConstruction, M);

    cout<<"[Index Building Begins]"<<endl;
    
    index->buildIndexFromBinaryData(dataset,dataset_size);

    /* 将训练权重的代码放置在这里 */
    /* code */

    
    index->write();

    cout << "[Index Building Time] "<< index->time_stats->index_building_time<<"(sec)"<<endl;

    }
    else if(mode==1){

        Index * index = Index::Read(index_path, mode); 
        
        QueryEngine * queryengine = new QueryEngine(query_dataset, query_dataset_size, 
                                                    groundtruth_dataset ,groundtruth_top_k, groundtruth_dataset_size,
                                                    learn_dataset, learn_dataset_size, learn_groundtruth_dataset,
                                                    dataset,
                                                    index, ef, nprobes, parallel, nworker, flatt, k, ep, model_file, zero_edge_pass_ratio); 

        queryengine->TrainWeightByLearnDataset(ep, k, mode);

        queryengine->queryBinaryFile(k, mode, thres_probability, μ, T);
        cout << "[Querying Time] "<< index->time_stats->querying_time <<"(sec)"<<endl;  
        cout << "[QPS] "<< query_dataset_size*1.0/index->time_stats->querying_time <<endl;  


        double averageRecall = queryengine->calculateAverageRecall();
        cout<<"[Querying Average Recall] "<< averageRecall <<endl;


        delete index;
        delete queryengine;

    }else if(mode==2){ 

        Hercules * hercules= new Hercules(dataset,dataset_size,index_path, 
                                        time_series_size, leaf_size,
                                        query_dataset, query_dataset_size,
			                            groundtruth_dataset, groundtruth_dataset_size, groundtruth_top_k,
                                        efConstruction, M);

                                        
       hercules->buildIndexTree();

	   hercules->generateAllFiles();
       
    }
    return 0;
}
