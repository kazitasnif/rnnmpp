#ifndef INET_H
#define INET_H

#include "dense_matrix.h"
#include "linear_param.h"
#include "const_scalar_param.h"
#include "nngraph.h"
#include "param_layer.h"
#include "input_layer.h"
#include "c_add_layer.h"
#include "c_mul_layer.h"
#include "fmt/format.h"
#include "fmt/printf.h"
#include "relu_layer.h"
#include "model.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "sigmoid_layer.h"
#include "classnll_criterion_layer.h"
#include "config.h"
#include "data_loader.h"
#include "err_cnt_criterion_layer.h"
#include "intensity_nll_criterion_layer.h"
#include "dur_pred_layer.h"
#include "learner.h"
#include <limits>

template<MatMode mode, typename Dtype>
class INet
{
public:
	INet(IEventTimeLoader<mode>* _etloader)
	{
        this->etloader = _etloader;
        initialized = false;
        g_last_hidden_train = new DenseMat<mode, Dtype>();
        g_last_hidden_test = new DenseMat<mode, Dtype>();
	InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label, g_value_input, g_value_label);
        learner = new MomentumSGDLearner<mode, Dtype>(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
	}

    void Setup()
    {
        InitParamDict();

        InitNet(net_train, model.all_params, cfg::bptt);
        InitNet(net_test, model.all_params, 1);
        initialized = true;
    }
    /*llh*/

    double llh(DataLoader<TEST>* dl, std::map<std::string, Dtype>& llh_map, unsigned seq_num){
        double ret = 0;
        auto& last_hidden_test = g_last_hidden_test->DenseDerived();
        last_hidden_test.Zeros(1, cfg::n_hidden);
	//LinkTestData(); 
	llh_map.clear();
        dl -> BeforeForwardSeq(seq_num);
	//std::cerr << "after beforeForwardSeq" << std::endl;
		
	//InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label, g_value_input, g_value_label);
	/*using the NextBatch code for forwarding a sequence with batch size = 1 */  
	unsigned iter = 0;
	/*for(size_t i = 0; i < dl -> event_sequences.size(); i++){
	 std::cerr << dl -> event_sequences[i].size() << " ";
	}
	std::cerr << std::endl;
	std::cerr << "seq size: " << dl -> event_sequences[seq_num].size() << std::endl;*/
	while (dl->ForwardSeq(etloader, 
                                  g_last_hidden_test, 
                                  g_event_input[0], 
                                  g_time_input[0], 
                                  g_value_input[0],
				  g_event_label[0], 
                                  g_time_label[0],
				  g_value_label[0]))
        {                        
            //std::cerr << "loaded batch in llh" << std::endl;
	    //g_time_label[0] -> Print2Screen();
	    //g_time_input[0] -> Print2Screen();
	    iter++;
	    //std::cerr << "iter: " << iter << std::endl;
	    net_test.FeedForward(test_dict, TEST);
	    //std::cerr << "feed forward" << std::endl;
            auto loss_map = net_test.GetLoss();
	    //std::cerr << "loss map in llh" << std::endl;
            for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
            {
               
		if (llh_map.count(it->first) == 0)
                    llh_map[it->first] = 0.0;
		//std::cerr << "debugging inside evaluate dataset" << std::endl;
		//std::cerr << it -> second << std::endl;
                ret += -1.0 * it -> second;
		llh_map[it->first] += -1.0 * it->second;
            }
	    //std::cerr << "ret: " << ret << std::endl;
            if (cfg::bptt > 1)
                net_test.GetState("recurrent_hidden_0", last_hidden_test);            
        }
	
      	//std::cerr << "ret: " << ret << std::endl;  	
	//InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label, g_value_input, g_value_label);
      	return ret;

    }
    void CalculateLLH(const char* prefix, DataLoader<TEST>* dataset, bool writeTofile){
	
	std::map<std::string, Dtype> llh_map;
	std::vector<double> llh_vec;
	
	for(size_t i = 0; i < dataset -> time_sequences.size(); i++){
	  llh_vec.push_back(llh(dataset, llh_map, i));
	}
	std::cout << "llh vec size: " << llh_vec.size() << std::endl;
	std::cout << llh_vec[0] << " " << llh_vec[1] << std::endl;
	FILE* fid = nullptr;
	if (writeTofile){
            fid = fopen(fmt::sprintf("%s/%s.txt", cfg::save_dir, prefix).c_str(), "w");
     	    for(size_t i = 0; i < llh_vec.size(); i++) fprintf(fid, "%f\n", llh_vec[i]);
	    //fprintf(fid, "\n");
	}
	fclose(fid);
	std::cout << std::endl;

    }
    void EvaluateDataset(const char* prefix, DataLoader<TEST>* dataset, bool save_prediction, std::map<std::string, Dtype>& test_loss_map)
    {
        auto& last_hidden_test = g_last_hidden_test->DenseDerived();
        last_hidden_test.Zeros(dataset->batch_size, cfg::n_hidden); 

        dataset->StartNewEpoch();
                
        test_loss_map.clear();
        FILE* fid = nullptr;
        if (save_prediction)
            fid = fopen(fmt::sprintf("%s/%s_pred_iter_%d.txt", cfg::save_dir, prefix, cfg::iter).c_str(), "w");
       	std::cerr << "file open" << std::endl; 
        size_t called_nextbatch = 0;
	while (dataset->NextBatch(etloader, 
                                  g_last_hidden_test, 
                                  g_event_input[0], 
                                  g_time_input[0], 
                                  g_value_input[0],
				  g_event_label[0], 
                                  g_time_label[0],
				  g_value_label[0]))
        {                        
            //std::cerr << "loaded batch" << std::endl;
	    //g_event_input[0] -> Print2Screen();
	   // g_value_input[0] -> Print2Screen();
	    //g_time_input[0] -> Print2Screen();
	    called_nextbatch++;
	    net_test.FeedForward(test_dict, TEST);
            auto loss_map = net_test.GetLoss();
            for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
            {
                if (test_loss_map.count(it->first) == 0)
                    test_loss_map[it->first] = 0.0;
		//std::cerr << "debugging inside evaluate dataset" << std::endl;
		//std::cerr << it -> first << " " << it -> second << std::endl;
                test_loss_map[it->first] += it->second;
		//time_input[0] -> Print2Screen();
		/*
            	if(test_loss_map[it -> first] >= std::numeric_limits<Dtype>::infinity() || test_loss_map[it->first] <= -1.0 * std::numeric_limits<Dtype>::infinity()){
		     
		     std::cerr << it -> first << " " << it -> second << std::endl;
              
		     g_event_input[0] -> Print2Screen();
	             g_value_input[0] -> Print2Screen();
	             g_time_input[0] -> Print2Screen();
	    	}*/
		
	    }
	    
	    /*for(auto it = test_loss_map.begin(); it != test_loss_map.end(); ++it){
		std::cerr << it -> first << " " << it -> second << std::endl;
		
	    }*/
            if (save_prediction)
                WriteTestBatch(fid);
            if (cfg::bptt > 1)
                net_test.GetState("recurrent_hidden_0", last_hidden_test);            
        
	    /*std::cerr << "g_last_hidden_test" << std::endl;
	    g_last_hidden_test -> Print2Screen();
	    std::cerr << "last_hidden_test" << std::endl;
	    last_hidden_test.Print2Screen();*/
	
	}
	std::cerr << "called nextbatch " << called_nextbatch << std::endl;
        if (save_prediction) fclose(fid);
    	}
    
	void MainLoop()
	{
	/*
	for(size_t i = 0; i <  train_data -> time_sequences.size(); i++){
	  for(size_t j = 0; j < train_data -> time_sequences[i].size(); j++){
	    std::cerr << train_data -> time_sequences[i][j] << " ";
	  }
	  std::cerr << std::endl;
	}*/
        if (!initialized)
            Setup();

		long long max_iter = (long long)cfg::max_epoch;
    	int init_iter = cfg::iter;
    
    	if (init_iter > 0)
    	{
            std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
            model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
    	}
        			
    	LinkTrainData();
    	LinkTestData();

        auto& last_hidden_train = g_last_hidden_train->DenseDerived();         
    	last_hidden_train.Zeros(cfg::batch_size, cfg::n_hidden);
    	std::map<std::string, Dtype> test_loss_map;
	std::map<std::string, Dtype> llh_map;
        	
       	train_data -> StartNewEpoch(); /*not in the original codebase*/ 
    	for (; cfg::iter <= max_iter; ++cfg::iter)
    	{
        	if (cfg::iter % cfg::test_interval == 0)
	        {
    	        std::cerr << "testing" << std::endl;
        	    
                EvaluateDataset("test", test_data, true, test_loss_map);
		//std::cerr << "after evaluate dataset" << std::endl;
		for(auto it = test_loss_map.begin(); it != test_loss_map.end(); ++it){
		std::cerr << it -> first << " " << it -> second << std::endl;
		
	    	}
           
		PrintTestResults(test_data, test_loss_map);
		/*for(unsigned i = 0; i < test_data -> event_sequences.size(); i++){
		  std::cout << "llh: " << llh(test_data, llh_map, i) << " ";
		} */
		std::cout << std::endl;
                if (cfg::has_eval)
                {
                    EvaluateDataset("val", val_data, cfg::save_eval, test_loss_map);
                    PrintTestResults(val_data, test_loss_map);    
                }
        	}
        
        	if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
        	{
            	std::cerr << fmt::sprintf("saving model for iter = %d", cfg::iter) << std::endl;
                model.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
        	}
        	train_data->NextBpttBatch(etloader, 
                                      cfg::bptt, 
            	                      g_last_hidden_train, 
                	                  g_event_input, 
                    	              g_time_input,
				      g_value_input, 
                        	          g_event_label, 
                            	      g_time_label,
				      g_value_label);
       		
		//g_event_input[0] -> Print2Screen(); 
        	//g_time_input[0] -> Print2Screen();
		net_train.FeedForward(train_dict, TRAIN);
        	auto loss_map = net_train.GetLoss();
            	/*std::cerr << "train loss" << std::endl;
		for (auto it = loss_map.begin(); it != loss_map.end(); ++it){
		  std::cerr << it -> first << " " << it -> second << std::endl;
		}*/	
	    if (cfg::bptt > 1 && cfg::use_history)
            {
                net_train.GetState(fmt::sprintf("recurrent_hidden_%d", cfg::bptt - 1), last_hidden_train);
            }

            net_train.BackPropagation();
            learner->Update();   

        	if (cfg::iter % cfg::report_interval == 0)
        	{
        		PrintTrainBatchResults(loss_map);
        	}
		if(cfg::iter == max_iter){
		  CalculateLLH("llh_test_survived", test_data, true);
		}
    	}
	
	}

	NNGraph<mode, Dtype> net_train, net_test;
    Model<mode, Dtype> model;
    MomentumSGDLearner<mode, Dtype>* learner;

	std::vector< IMatrix<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label, g_value_input, g_value_label;	
	std::map<std::string, IMatrix<mode, Dtype>* > train_dict, test_dict;
    IEventTimeLoader<mode>* etloader;

    bool initialized;
    void InitNet(NNGraph<mode, Dtype>& gnn, 
                 std::map< std::string, IParam<mode, Dtype>* >& param_dict, 
                 unsigned n_unfold)
    {
        auto* last_hidden_layer = cl< InputLayer >("last_hidden", gnn, {});

        for (unsigned i = 0; i < n_unfold; ++i)
        {
            auto* new_hidden = AddNetBlocks(i, gnn, last_hidden_layer, param_dict);
            last_hidden_layer = new_hidden;
        }
    }    
    IMatrix<mode, Dtype>* g_last_hidden_train, *g_last_hidden_test;

    virtual void WriteTestBatch(FILE* fid) = 0;
	virtual void LinkTrainData() = 0;
	virtual void LinkTestData() = 0;
	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) = 0;
	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) = 0;

	virtual void InitParamDict() = 0;
	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  NNGraph<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict) = 0;
};

#endif
