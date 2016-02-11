#ifndef TIME_NET_H
#define TIME_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class TimeNet : public INet<mode, Dtype>
{
public:
	TimeNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

	virtual void LinkTrainData() override
	{
    	this->train_feat["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
        	this->train_feat[fmt::sprintf("time_input_%d", i)] = this->g_time_input[i];
        	this->train_label[fmt::sprintf("mse_%d", i)] = this->g_time_label[i];
        	this->train_label[fmt::sprintf("mae_%d", i)] = this->g_time_label[i];
        	if (cfg::loss_type == LossType::EXP)
        		this->train_label[fmt::sprintf("expnll_%d", i)] = this->g_time_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_feat["last_hidden"] = this->g_last_hidden_test;
		this->test_feat["time_input_0"] = this->g_time_input[0];
		this->test_label["mse_0"] = this->g_time_label[0];
		this->test_label["mae_0"] = this->g_time_label[0];
		if (cfg::loss_type == LossType::EXP)
			this->test_label["expnll_0"] = this->g_time_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) override
	{
		Dtype rmse = 0.0, mae = 0.0, expnll = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
            rmse += loss_map[fmt::sprintf("mse_%d", i)];  
            if (cfg::loss_type == LossType::EXP)
            	expnll += loss_map[fmt::sprintf("expnll_%d", i)];  
        }
        rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
		mae /= cfg::bptt * cfg::batch_size;
		expnll /= cfg::bptt * cfg::batch_size;
		std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f", cfg::iter, mae, rmse);
		if (cfg::loss_type == LossType::EXP)
			std::cerr << fmt::sprintf("\texpnll: %.4f", expnll);
		std::cerr << std::endl;
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) override
	{
		Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"];
		rmse = sqrt(rmse / dataset->num_samples);
		mae /= dataset->num_samples;
		std::cerr << fmt::sprintf("test_mae: %.4f\t test_rmse: %.4f", mae, rmse);
		if (cfg::loss_type == LossType::EXP)
			std::cerr << fmt::sprintf("\texpnll: %.4f", loss_map["expnll_0"] / dataset->num_samples);
		std::cerr << std::endl;
	}

	virtual void InitParamDict() override
	{
		this->param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);

    	if (cfg::gru)
        {
            this->param_dict["w_h2update"] = new LinearParam<mode, Dtype>("w_h2update", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_time2update"] = new LinearParam<mode, Dtype>("w_time2update", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_h2reset"] = new LinearParam<mode, Dtype>("w_h2reset", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_time2reset"] = new LinearParam<mode, Dtype>("w_time2reset", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);                        
        }

        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            this->param_dict["w_hidden2"] = new LinearParam<mode, Dtype>("w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }

        this->param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", hidden_size, 1, 0, cfg::w_scale);
	}

    virtual ILayer<mode, Dtype>* AddRecur(std::string name, 
                                          GraphNN<mode, Dtype>& gnn,
                                          ILayer<mode, Dtype> *last_hidden_layer, 
                                          ILayer<mode, Dtype>* time_feat, 
                                          IParam<mode, Dtype>* h2h, 
                                          IParam<mode, Dtype>* t2h)
    {
        auto* hidden_layer = new NodeLayer<mode, Dtype>(name);
        hidden_layer->AddParam(time_feat->name, t2h, GraphAtt::NODE); 
        hidden_layer->AddParam(last_hidden_layer->name, h2h, GraphAtt::NODE); 

        gnn.AddEdge(time_feat, hidden_layer);
        gnn.AddEdge(last_hidden_layer, hidden_layer);
        return hidden_layer;
    }	

    virtual ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        auto* hidden_layer = AddRecur(fmt::sprintf("hidden_%d", time_step), 
                                      gnn, last_hidden_layer, time_feat,
                                      param_dict["w_h2h"], param_dict["w_time2h"]);        

        auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("recurrent_hidden_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(hidden_layer, relu_hidden_layer);

        return relu_hidden_layer;
    }

    virtual ILayer<mode, Dtype>* AddGRULayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        // local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* update_linear = AddRecur(fmt::sprintf("update_linear_%d", time_step), 
                                       gnn, last_hidden_layer, time_feat, 
                                       param_dict["w_h2update"], param_dict["w_time2update"]); 
        auto* update_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("update_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(update_linear, update_gate);

        // local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* reset_linear = AddRecur(fmt::sprintf("reset_linear_%d", time_step), 
                                      gnn, last_hidden_layer, time_feat, 
                                      param_dict["w_h2reset"], param_dict["w_time2reset"]);         
        auto* reset_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("reset_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(reset_linear, reset_gate);

        // local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        auto* gated_hidden = new PairMulLayer<mode, Dtype>(fmt::sprintf("gated_hidden_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(reset_gate, gated_hidden);
        gnn.AddEdge(last_hidden_layer, gated_hidden);

        // local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        auto* hidden_candidate_linear = AddRecur(fmt::sprintf("hidden_candidate_linear%d", time_step), 
                                          gnn, gated_hidden, time_feat,
                                          param_dict["w_h2h"], param_dict["w_time2h"]);
        auto* hidden_candidate = new ReLULayer<mode, Dtype>(fmt::sprintf("hidden_candidate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(hidden_candidate_linear, hidden_candidate);

        // local zh = nn.CMulTable()({update_gate, hidden_candidate})
        auto* zh = new PairMulLayer<mode, Dtype>(fmt::sprintf("zh_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(hidden_candidate, zh);
        gnn.AddEdge(update_gate, zh);

        // nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate))
        auto* z_prev_h = new ConstTransLayer<mode, Dtype>(fmt::sprintf("z_prev_h_%d", time_step), GraphAtt::NODE, -1, 1);
        gnn.AddEdge(update_gate, z_prev_h);

        // (1 - update_gate) * prev_h
        auto* zhm1 = new PairMulLayer<mode, Dtype>(fmt::sprintf("zhm1_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(z_prev_h, zhm1);
        gnn.AddEdge(last_hidden_layer, zhm1);

        // local next_h = nn.CAddTable()({zh, zhm1})
        auto* next_h = new NodeGatherLayer<mode, Dtype>(fmt::sprintf("recurrent_hidden_%d", time_step));
        gnn.AddEdge(zh, next_h);
        gnn.AddEdge(zhm1, next_h);

        return next_h;
    }

	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  GraphNN<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict) override
	{
		gnn.AddLayer(last_hidden_layer);
    
		auto* time_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step), GraphAtt::NODE);

		ILayer<mode, Dtype>* recurrent_output = nullptr;
        if (cfg::gru)
        {
            recurrent_output = AddGRULayer(time_step, gnn, last_hidden_layer, time_input_layer, param_dict);
        } else
            recurrent_output = AddRNNLayer(time_step, gnn, last_hidden_layer, time_input_layer, param_dict);

        auto* top_hidden = recurrent_output;
        if (cfg::n_h2)
        {
            auto* hidden_2 = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("hidden_2_%d", time_step), param_dict["w_hidden2"], GraphAtt::NODE);
            gnn.AddEdge(recurrent_output, hidden_2);
            auto* relu_2 = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_h2_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
            gnn.AddEdge(hidden_2, relu_2);
            top_hidden = relu_2;
        }

	    auto* time_out_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"], GraphAtt::NODE); 
	    gnn.AddEdge(top_hidden, time_out_layer);
	    
	    //auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);    
	    //gnn.AddEdge(time_out_layer, exp_layer);
    	auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step),  
    															 cfg::loss_type == LossType::MSE ? PropErr::T : PropErr::N);
    	auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step),  PropErr::N);

    	gnn.AddEdge(time_out_layer, mse_criterion);
    	gnn.AddEdge(time_out_layer, mae_criterion);

    	if (cfg::loss_type == LossType::EXP)
    	{
    		assert(false);
    		auto* expnll_criterion = new ExpNLLCriterionLayer<mode, Dtype>(fmt::sprintf("expnll_%d", time_step));
    		gnn.AddEdge(time_out_layer, expnll_criterion); 
    	}

    	return recurrent_output; 
	}

	virtual void WriteTestBatch(FILE* fid) override
	{
		this->net_test.GetDenseNodeState("time_out_0", buf);		
		buf2.CopyFrom(this->g_time_label[0]->node_states->DenseDerived());
        for (size_t i = 0; i < buf.rows; ++i)
            fprintf(fid, "%.6f %.6f\n",  buf.data[i], buf2.data[i]);
	}

	DenseMat<CPU, Dtype> buf, buf2;
};

#endif
