#ifndef JOINT_VALUE_NET_H
#define JOINT_VALUE_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class JointValueNet : public INet<mode, Dtype>
{
public:

	JointValueNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

	virtual void LinkTrainData() override 
	{
    	this->train_dict["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
    		this->train_dict[fmt::sprintf("event_input_%d", i)] = this->g_event_input[i];
        	this->train_dict[fmt::sprintf("time_input_%d", i)] = this->g_time_input[i];
            	this->train_dict[fmt::sprintf("value_input_%d", i)] = this->g_value_input[i];
	    this->train_dict[fmt::sprintf("event_%d", i)] = this->g_event_label[i];
            this->train_dict[fmt::sprintf("dur_%d", i)] = this->g_time_label[i];
	    this -> train_dict[fmt::sprintf("value_%d", i)] = this->g_value_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_dict["last_hidden"] = this->g_last_hidden_test;
		this->test_dict["event_input_0"] = this->g_event_input[0];
		this->test_dict["time_input_0"] = this->g_time_input[0];
		this->test_dict["value_input_0"] = this->g_value_input[0];
		this->test_dict["dur_0"] = this->g_time_label[0];
		this->test_dict["event_0"] = this->g_event_label[0];
	        this->test_dict["value_0"] = this->g_value_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = 0.0, mae = 0.0, nll = 0.0, err_cnt = 0.0, intnll = 0.0;
		Dtype val_mae = 0.0, val_rmse = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
	    std:: cerr << loss_map[fmt::sprintf("mae_%d", i)] << std::endl;
	    std:: cerr << loss_map[fmt::sprintf("val_mae_%d", i)] << std::endl;
	    std::cerr << "mae: " << mae << std::endl;
            rmse += loss_map[fmt::sprintf("mse_%d", i)];
	    std::cerr << "rmse: " << rmse << std::endl;
            val_mae += loss_map[fmt::sprintf("val_mae_%d", i)];
            val_rmse += loss_map[fmt::sprintf("val_mse_%d", i)];
        	
	    nll += loss_map[fmt::sprintf("nll_%d", i)]; 
            err_cnt += loss_map[fmt::sprintf("err_cnt_%d", i)]; 
            if (cfg::loss_type == LossType::INTENSITY)
                intnll +=  loss_map[fmt::sprintf("intnll_%d", i)]; 
        }
        rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
		mae /= cfg::bptt * cfg::batch_size;
	val_rmse = sqrt(val_rmse / cfg::bptt / cfg::batch_size);
		val_mae /= cfg::bptt * cfg::batch_size;
		
	nll /= cfg::bptt * train_data->batch_size;
        err_cnt /= cfg::bptt * train_data->batch_size;
        intnll /= cfg::bptt * cfg::batch_size;
        std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f\tvmae: %.4f\tvrmse: %.4f\tnll: %.4f\terr_rate: %.4f", cfg::iter, mae, rmse, val_mae, val_rmse,  nll, err_cnt);
        if (cfg::loss_type == LossType::INTENSITY)
            std::cerr << fmt::sprintf("\tintnll: %.4f", intnll);
        std::cerr << std::endl;
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"], nll = loss_map["nll_0"];
		std::cout << "rmse: " << rmse << std::endl;
		std::cout << "mae: " << mae << std::endl;
		if(cfg::loss_type == LossType::INTENSITY){
		  std::cout << "intnll" << loss_map["intnll_0"] << std::endl;
		}
		std::cout << "nll: " << nll << std::endl;
	 
		Dtype val_rmse = loss_map["val_mse_0"], val_mae = loss_map["val_mae_0"];
		rmse = sqrt(rmse / dataset->num_samples);
		val_rmse = sqrt(val_rmse / dataset->num_samples);
		val_mae /= dataset -> num_samples;
		mae /= dataset->num_samples;
		nll /= dataset->num_samples;
		
        Dtype err_cnt = loss_map["err_cnt_0"] / dataset->num_samples;
        std::cerr << fmt::sprintf("test_mae: %.6f\ttest_rmse: %.6f\ttest_nll: %.4f\ttest_err_rate: %.4f\ttest_val_rmse: %.4f\ttest_val_mae: %.4f", mae, rmse, nll, err_cnt,val_rmse,val_mae);

        if (cfg::loss_type == LossType::INTENSITY)
        {
            std::cerr << fmt::sprintf("\ttest_intnll: %.6f", loss_map["intnll_0"] / dataset->num_samples);
        }        
        std::cerr << std::endl;        
	}

	virtual void InitParamDict() 
	{
		add_diff< LinearParam >(this->model, "w_embed", train_data->num_events, cfg::n_embed, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
		add_diff< LinearParam >(this->model, "w_time2h", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
		add_diff<LinearParam>(this->model, "w_value2h", cfg::value_dim, cfg::n_hidden, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
        
        if (cfg::gru)
        {
            add_const< ConstScalarParam >(this->model, "const_scalar", -1, 1); 
            add_diff< LinearParam >(this->model, "w_h2update", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            add_diff< LinearParam >(this->model, "w_event2update", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
            add_diff< LinearParam >(this->model, "w_time2update", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
	    add_diff< LinearParam >(this->model, "w_value2update", cfg::value_dim, cfg::n_hidden, 0, cfg::w_scale);
	    add_diff< LinearParam >(this->model, "w_h2reset", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            add_diff< LinearParam >(this->model, "w_event2reset", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
            add_diff< LinearParam >(this->model, "w_time2reset", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);  
	    add_diff< LinearParam >(this->model, "w_value2reset", cfg::value_dim, cfg::n_hidden, 0, cfg::w_scale);  
	                      
        }
        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            add_diff< LinearParam >(this->model, "w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }
        add_diff< LinearParam >(this->model, "w_event_out", hidden_size, train_data->num_events, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_time_out", hidden_size, 1, 0, cfg::w_scale);
        add_diff< LinearParam> (this->model, "w_value_out", hidden_size, 1, 0, cfg::w_scale);

	if (cfg::loss_type == LossType::INTENSITY)
            add_diff< LinearParam >(this->model, "w_lambdat", 1, 1, 0, cfg::w_scale, BiasOption::NONE); 
	}

    virtual ILayer<mode, Dtype>* AddRecur(std::string name, 
                                          NNGraph<mode, Dtype>& gnn,
                                          ILayer<mode, Dtype> *last_hidden_layer, 
                                          ILayer<mode, Dtype>* event_feat, 
                                          ILayer<mode, Dtype>* time_feat, 
                                          ILayer<mode, Dtype>* value_feat,
					  IParam<mode, Dtype>* h2h, 
                                          IParam<mode, Dtype>* e2h,
                                          IParam<mode, Dtype>* t2h,
					  IParam<mode, Dtype>* v2h)
    {
        return cl< ParamLayer >(gnn, {value_feat, time_feat, event_feat, last_hidden_layer}, {v2h, t2h, e2h, h2h}); 
    }

    virtual ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
                                             NNGraph<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             ILayer<mode, Dtype>* value_feat,
					     std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        auto* hidden_layer = AddRecur(fmt::sprintf("hidden_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat, time_feat, value_feat,
                                      param_dict["w_h2h"], param_dict["w_event2h"], param_dict["w_time2h"],
				      param_dict["w_value2h"]);        
        return cl< ReLULayer >(fmt::sprintf("recurrent_hidden_%d", time_step), gnn, {hidden_layer});
    }

    virtual ILayer<mode, Dtype>* AddGRULayer(int time_step, 
                                             NNGraph<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             ILayer<mode, Dtype>* time_feat, 
					     ILayer<mode, Dtype>* value_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        // local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* update_linear = AddRecur(fmt::sprintf("update_linear_%d", time_step), 
                                       gnn, last_hidden_layer, event_feat, time_feat, value_feat,
                                       param_dict["w_h2update"], param_dict["w_event2update"], param_dict["w_time2update"],
				       param_dict["w_value2update"]); 
        auto* update_gate = cl<SigmoidLayer>(gnn, {update_linear});

        // local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* reset_linear = AddRecur(fmt::sprintf("reset_linear_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat, time_feat, value_feat,
                                      param_dict["w_h2reset"], param_dict["w_event2reset"], param_dict["w_time2reset"],
				      param_dict["w_value2reset"] );         
        auto* reset_gate = cl<SigmoidLayer>(gnn, {reset_linear});

        // local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        auto* gated_hidden = cl<CMulLayer>(gnn, {reset_gate, last_hidden_layer});

        // local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        auto* hidden_candidate_linear = AddRecur(fmt::sprintf("hidden_candidate_linear%d", time_step), 
                                          gnn, gated_hidden, event_feat, time_feat, value_feat,
                                          param_dict["w_h2h"], param_dict["w_event2h"], param_dict["w_time2h"], 
					  param_dict["w_value2h"]);
        auto* hidden_candidate = cl< ReLULayer >(gnn, {hidden_candidate_linear});

        // local zh = nn.CMulTable()({update_gate, hidden_candidate})
        auto* zh = cl< CMulLayer >(gnn, {hidden_candidate, update_gate});

        // nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate))
        auto* z_prev_h = cl< ParamLayer >(gnn, {update_gate}, {param_dict["const_scalar"]});

        // (1 - update_gate) * prev_h
        auto* zhm1 = cl< CMulLayer >(gnn, {z_prev_h, last_hidden_layer});

        // local next_h = nn.CAddTable()({zh, zhm1})
        auto* next_h = cl< CAddLayer >(fmt::sprintf("recurrent_hidden_%d", time_step), gnn, {zh, zhm1});

        return next_h;
    }

	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  NNGraph<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict)
	{
        auto* time_input_layer = cl< InputLayer >(fmt::sprintf("time_input_%d", time_step), gnn, {});
        auto* event_input_layer = cl< InputLayer >(fmt::sprintf("event_input_%d", time_step), gnn, {});
	auto* value_input_layer = cl< InputLayer >(fmt::sprintf("value_input_%d", time_step), gnn, {});

        auto* dur_label_layer = cl< InputLayer >(fmt::sprintf("dur_%d", time_step), gnn, {});
        auto* event_label_layer = cl< InputLayer >(fmt::sprintf("event_%d", time_step), gnn, {});
        auto* value_label_layer = cl< InputLayer >(fmt::sprintf("value_%d", time_step), gnn, {});
	
    	auto* embed_layer = cl< ParamLayer >(gnn, {event_input_layer}, {param_dict["w_embed"]});
    	auto* relu_embed_layer = cl< ReLULayer >(gnn, {embed_layer});

    	ILayer<mode, Dtype>* recurrent_output = nullptr;
        if (cfg::gru)
        {
            recurrent_output = AddGRULayer(time_step, gnn, last_hidden_layer, relu_embed_layer, time_input_layer, value_input_layer, param_dict);
        } else
            recurrent_output = AddRNNLayer(time_step, gnn, last_hidden_layer, relu_embed_layer, time_input_layer, value_input_layer, param_dict);
    	
        auto* top_hidden = recurrent_output;
        if (cfg::n_h2)
        {
            auto* hidden_2 = cl< ParamLayer >(gnn, {recurrent_output}, {param_dict["w_hidden2"]});
            auto* relu_2 = cl< ReLULayer >(gnn, {hidden_2});
            top_hidden = relu_2;
        } 

        auto* event_output_layer = cl< ParamLayer >(fmt::sprintf("event_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_event_out"]}); 

        auto* time_out_layer = cl< ParamLayer >(fmt::sprintf("time_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_time_out"]});
		
        auto* value_out_layer = cl< ParamLayer >(fmt::sprintf("value_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_value_out"]});


        cl< ClassNLLCriterionLayer >(fmt::sprintf("nll_%d", time_step), gnn, {event_output_layer, event_label_layer}, true);
        cl< ErrCntCriterionLayer >(fmt::sprintf("err_cnt_%d", time_step), gnn, {event_output_layer, event_label_layer});

        if (cfg::loss_type == LossType::MSE)
        {
            cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), gnn, {time_out_layer, dur_label_layer}, cfg::lambda);
            cl< ABSCriterionLayer >(fmt::sprintf("mae_%d", time_step), gnn, {time_out_layer, dur_label_layer}, PropErr::N);
        }
        if (cfg::loss_type == LossType::INTENSITY)
        {  
	    std::cerr << "intensity losstype" << std::endl; 
            LinearParam<mode, Dtype>* w = dynamic_cast<LinearParam<mode, Dtype>*>(param_dict["w_lambdat"]);
            cl< IntensityNllCriterionLayer >(fmt::sprintf("intnll_%d", time_step), 
                                             gnn, 
                                             {time_out_layer, dur_label_layer}, 
                                             w, 
                                             cfg::lambda);
            auto* dur_pred = cl< DurPredLayer >(fmt::sprintf("dur_pred_%d", time_step), 
                                                gnn, 
                                                {time_out_layer}, 
                                                w);
            cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), gnn, {dur_pred, dur_label_layer}, PropErr::N);
            cl< ABSCriterionLayer >(fmt::sprintf("mae_%d", time_step), gnn, {dur_pred, dur_label_layer}, PropErr::N);
        }
        
	cl< MSECriterionLayer >(fmt::sprintf("val_mse_%d", time_step), gnn, {value_out_layer, value_label_layer}, PropErr::T);
        cl< ABSCriterionLayer >(fmt::sprintf("val_mae_%d", time_step), gnn, {value_out_layer, value_label_layer}, PropErr::N);
      
		return recurrent_output; 
	}

    virtual void WriteTestBatch(FILE* fid) override
    {
        /*
        this->net_test.GetDenseNodeState("time_out_0", time_pred);
        this->net_test.GetDenseNodeState("event_out_0", event_pred);
        for (size_t i = 0; i < time_pred.rows; ++i)
        {
            fprintf(fid, "%.6f ", time_pred.data[i]);
            int pred = 0; 
            Dtype best = event_pred.data[i * event_pred.cols];
            for (size_t j = 1; j < event_pred.cols; ++j)
                if (event_pred.data[i * event_pred.cols + j] > best)
                {
                    best = event_pred.data[i * event_pred.cols + j]; 
                    pred = j;
                }
            fprintf(fid, "%d\n", pred);
        }*/
    }

    DenseMat<CPU, Dtype> time_pred, event_pred, value_pred;
};

#endif
