#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <set>
#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "mkl_helper.h"
#include "cuda_helper.h"

template<MatMode mode>
class IEventTimeLoader;

class IDataLoader
{
public:
   
    IDataLoader(size_t _num_events, size_t _batch_size) : num_events(_num_events), batch_size(_batch_size)
    {
        event_sequences.clear();
        time_sequences.clear();
        value_sequences.clear();
	time_label_sequences.clear();
	value_label_sequences.clear();
        cursors.resize(batch_size);
        index_pool.clear();
    } 
    

    inline void InsertSequence(int* event_seq, Dtype* time_seq, Dtype* time_label, int seq_len)
    {
        num_samples += seq_len - 1;
        InsertSequence(event_seq, event_sequences, seq_len);
        InsertSequence(time_seq, time_sequences, seq_len); 
        InsertSequence(time_label, time_label_sequences, seq_len - 1); 
    }
    


    inline void InsertSequence(int* event_seq, Dtype* time_seq, Dtype* time_label, Dtype* value_seq 
	, int seq_len)
    {
        num_samples += seq_len - 1;
        InsertSequence(event_seq, event_sequences, seq_len);
        InsertSequence(time_seq, time_sequences, seq_len); 
        InsertSequence(time_label, time_label_sequences, seq_len - 1); 
	InsertSequence(value_seq, value_sequences, seq_len);
    }
    
    virtual void StartNewEpoch()
    {
        initialized = true;
        if (index_pool.size() != event_sequences.size())
        {
            index_pool.clear();
            assert(event_sequences.size() == time_sequences.size()); 
            for (unsigned i = 0; i < event_sequences.size(); ++i)
            {
                index_pool.push_back(i);                     
            }
        }
        for (unsigned i = 0; i < batch_size; ++i)
        {
            cursors[i].first = index_pool.front();
            cursors[i].second = 0;
            index_pool.pop_front();
        }
    }
    virtual void BeforeForwardSeq(unsigned seq_num){
 	//index_pool.clear();
	cursors.resize(1);
	cursors[0].first = seq_num;
	cursors[0].second = 0;
    } 
    size_t num_samples, num_events, batch_size; 

    void ReloadSlot(unsigned batch_idx)    
    {
        index_pool.push_back(cursors[batch_idx].first); 
        cursors[batch_idx].first = index_pool.front();
        cursors[batch_idx].second = 0;
        index_pool.pop_front();
    }

    template<typename data_type>
    inline void InsertSequence(data_type* seq, std::vector< std::vector<data_type> >& sequences, int seq_len)
    {
        std::vector<data_type> cur_seq;
        cur_seq.clear();
        for (int i = 0; i < seq_len; ++i)
            cur_seq.push_back(seq[i]);
        sequences.push_back(cur_seq);
    }   

    void ReloadSlot(IMatrix<CPU, Dtype>* g_last_hidden, unsigned batch_idx)
    {
        auto& last_hidden = g_last_hidden->DenseDerived();
        memset(last_hidden.data + last_hidden.cols * batch_idx, 0, sizeof(Dtype) * last_hidden.cols);

        ReloadSlot(batch_idx);
    }    

    void ReloadSlot(IMatrix<GPU, Dtype>* g_last_hidden, unsigned batch_idx)
    {
        auto& last_hidden = g_last_hidden->DenseDerived();
        cudaMemset(last_hidden.data + last_hidden.cols * batch_idx, 0, sizeof(Dtype) * last_hidden.cols); 

        ReloadSlot(batch_idx);
    }
   
    bool initialized;
    std::vector< std::pair<unsigned, unsigned> > cursors;                 
    std::vector< std::vector<int> > event_sequences;
    std::vector< std::vector<Dtype> > time_sequences, time_label_sequences;
    std::vector< std::vector<Dtype> > value_sequences, value_label_sequences;
    std::deque< unsigned > index_pool;
};


template<Phase phase>
class DataLoader; 

template<>
class DataLoader<TRAIN> : public IDataLoader
{
public:

    DataLoader(unsigned _num_events, unsigned _batch_size) : IDataLoader(_num_events, _batch_size)
    {
        
    }
    
    template<MatMode mode>             
    inline void NextBpttBatch(IEventTimeLoader<mode>* etloader, 
                              int bptt, IMatrix<mode, Dtype>* g_last_hidden,
                              std::vector< IMatrix<mode, Dtype>* >& g_event_input,
                              std::vector< IMatrix<mode, Dtype>* >& g_time_input,
			      std::vector< IMatrix<mode, Dtype>* >& g_value_input, 
                              std::vector< IMatrix<mode, Dtype>* >& g_event_label,
                              std::vector< IMatrix<mode, Dtype>* >& g_time_label,
			      std::vector< IMatrix<mode, Dtype>* >& g_value_label)
    {
        if (!initialized)
            this->StartNewEpoch();

        for (unsigned i = 0; i < this->batch_size; ++i)
        {
            // need to load a new sequences                                   
            if (cursors[i].second + bptt >= event_sequences[cursors[i].first].size())
            {                              
                this->ReloadSlot(g_last_hidden, i); 
            }
        }
        for (int j = 0; j < bptt; ++j)
        {                                  
            etloader->LoadEvent(this, g_event_input[j], g_event_label[j], this->batch_size, j);                        
            etloader->LoadTime(this, g_time_input[j], g_time_label[j], this->batch_size, j);           
            if(cfg::has_value){
	      etloader->LoadValue(this, g_value_input[j], g_value_label[j], this->batch_size, j);           
	    }
	}        
        for (unsigned i = 0; i < this->batch_size; ++i)
            cursors[i].second += bptt;           
    }
}; 

template<>
class DataLoader<TEST> : public IDataLoader
{
public:
    
    DataLoader(unsigned _num_events, unsigned _batch_size) : IDataLoader(_num_events, _batch_size)
    {
        available.clear();
    }    
    
    template<MatMode mode>
    inline bool NextBatch(IEventTimeLoader<mode>* etloader, 
                          IMatrix<mode, Dtype>* g_last_hidden,
                          IMatrix<mode, Dtype>* g_event_input, 
                          IMatrix<mode, Dtype>* g_time_input, 
			  IMatrix<mode, Dtype>* g_value_input,
                          IMatrix<mode, Dtype>* g_event_label, 
                          IMatrix<mode, Dtype>* g_time_label,
			  IMatrix<mode, Dtype>* g_value_label)
    {
	//std::cerr << "inside nextbatch" << std::endl;
        if (!this->initialized)
            this->StartNewEpoch();
        unsigned delta_size = 0;                    
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            // need to load a new sequences                                   
            if (cursors[i].second + 1 >= event_sequences[cursors[i].first].size())
            {                
                if (index_pool.size() > 0 && available[index_pool.front()])
                {   
                    available[index_pool.front()] = false;                       
                    this->ReloadSlot(g_last_hidden, i);
                } else 
                    delta_size++;
            }
        }

        if (cur_batch_size == delta_size)
            return false;
       	//std::cerr << "before delta size" << std::endl; 
        if (delta_size)
        {
            auto& prev_hidden = g_last_hidden->DenseDerived();    
            if (cur_batch_size == batch_size) // insufficient for the first time
            {
                std::vector<unsigned> ordered; 
                for (unsigned i = 0; i < batch_size; ++i)
                    ordered.push_back(i);

                for (unsigned i = 0; i < batch_size - 1; ++i)
                    for (unsigned j = i + 1; j < batch_size; ++j)
                    {
                        if (cursors[j].second + 1 >= event_sequences[cursors[j].first].size())
                            continue;  // no need to move forward 

                        // if x is full, or y is longer than x
                        if (cursors[i].second + 1 >= event_sequences[cursors[i].first].size() || 
                            event_sequences[cursors[j].first].size() - cursors[j].second > 
                            event_sequences[cursors[i].first].size() - cursors[i].second )
                        {
                            unsigned tmp = ordered[i];
                            ordered[i] = ordered[j];
                            ordered[j] = tmp;
                            auto t = cursors[i];
                            cursors[i] = cursors[j];
                            cursors[j] = t;
                        }
                    }
                DenseMat<mode, Dtype> buf(batch_size - delta_size, prev_hidden.cols);

                for (unsigned i = 0; i < buf.rows; ++i)
                {
                    cudaMemcpy(buf.data + i * buf.cols, prev_hidden.data + ordered[i] * buf.cols, sizeof(Dtype) * buf.cols, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToDevice); 
                }  
                prev_hidden.CopyFrom(buf);     
            } else
                prev_hidden.Resize(cur_batch_size - delta_size, prev_hidden.cols);
            cur_batch_size -= delta_size;
        }
	//std::cerr<< " before loading events" << std::endl;
        etloader->LoadEvent(this, g_event_input, g_event_label, cur_batch_size, 0);
        etloader->LoadTime(this, g_time_input, g_time_label, cur_batch_size, 0);
        //std::cerr << "before loading value" << std::endl;
	if(cfg::has_value){
	  etloader->LoadValue(this, g_value_input, g_value_label, cur_batch_size, 0);
 	}
	//std::cerr << "after loading value" << std::endl;
	for (unsigned i = 0; i < cur_batch_size; ++i)
            cursors[i].second++;         
        return true;
    }
     
    template<MatMode mode>
    inline bool ForwardSeq(IEventTimeLoader<mode>* etloader, 
                          IMatrix<mode, Dtype>* g_last_hidden,
                          IMatrix<mode, Dtype>* g_event_input, 
                          IMatrix<mode, Dtype>* g_time_input, 
			  IMatrix<mode, Dtype>* g_value_input,
                          IMatrix<mode, Dtype>* g_event_label, 
                          IMatrix<mode, Dtype>* g_time_label,
			  IMatrix<mode, Dtype>* g_value_label
			  )
    {
            if (cursors[0].second + 1 >= event_sequences[cursors[0].first].size()) return false;
            //std::cerr<< " before loading events" << std::endl;
            etloader->LoadEvent(this, g_event_input, g_event_label, 1, 0);
            etloader->LoadTime(this, g_time_input, g_time_label, 1, 0);
            //std::cerr << "before loading value" << std::endl;
	    if(cfg::has_value){
	      etloader->LoadValue(this, g_value_input, g_value_label, 1, 0);
  	    }
	    //std::cerr << "after loading value" << std::endl;
            cursors[0].second++;         
            return true;
    }


    virtual void StartNewEpoch() override
    {                
        IDataLoader::StartNewEpoch();        
        if (available.size() != event_sequences.size())
            available.resize(event_sequences.size());
        
        for (unsigned i = 0; i < available.size(); ++i)
            available[i] = true;
	std::cout << available.size() << std::endl;
        assert(available.size() >= this->batch_size);
        
        for (unsigned i = 0; i < this->batch_size; ++i)
            available[cursors[i].first] = false;

        cur_batch_size = batch_size;        
    }
    /*virtual void BeforeForwardSeq(unsigned seq_num) override
    {
	this -> initialized = true;
        IDataLoader::BeforeForwardSeq(seq_num);
	if (available.size() != event_sequences.size())
            available.resize(event_sequences.size());
        
        for (unsigned i = 0; i < available.size(); ++i)
            available[i] = false;
	cur_batch_size = 1;	
    } */
protected:    
    unsigned cur_batch_size;
    std::vector<bool> available;             
}; 

template<MatMode mode>
class IEventTimeLoader
{
public:
    IEventTimeLoader() {}

    
    void LoadEvent(IDataLoader* d, IMatrix<mode, Dtype>* g_feat, IMatrix<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step)
    {
        auto& feat = g_feat->SparseDerived();
        auto& label = g_label->SparseDerived();
        
        this->event_feat_cpu.Resize(cur_batch_size, d->num_events);
        this->event_feat_cpu.ResizeSp(cur_batch_size, cur_batch_size + 1); 
        
        this->event_label_cpu.Resize(cur_batch_size, d->num_events);
        this->event_label_cpu.ResizeSp(cur_batch_size, cur_batch_size + 1);
        
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            this->event_feat_cpu.data->ptr[i] = i;
            this->event_feat_cpu.data->col_idx[i] = d->event_sequences[d->cursors[i].first][d->cursors[i].second + step]; 
            this->event_feat_cpu.data->val[i] = 1;
            
            this->event_label_cpu.data->ptr[i] = i;
            this->event_label_cpu.data->col_idx[i] = d->event_sequences[d->cursors[i].first][d->cursors[i].second + step + 1];
            this->event_label_cpu.data->val[i] = 1;                        
        }
        this->event_feat_cpu.data->ptr[cur_batch_size] = cur_batch_size;
        this->event_label_cpu.data->ptr[cur_batch_size] = cur_batch_size;
        
        feat.CopyFrom(this->event_feat_cpu);
        label.CopyFrom(this->event_label_cpu);
    } 
    void LoadValue(IDataLoader* d, IMatrix<mode, Dtype>* g_feat, IMatrix<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step)
    {
        auto& feat = g_feat->DenseDerived();
        auto& label = g_label->DenseDerived();
        
        this->value_feat_cpu.Resize(cur_batch_size, 1);
        this->value_label_cpu.Resize(cur_batch_size, 1);
        
	/*std::cerr << "value sequences size: " << d->value_sequences.size() << std::endl;
	std::cerr << "value label sequences size: " << d->value_label_sequences.size() << std::endl;

	for(size_t i = 0; i < d->value_sequences.size(); i++){
	  std::cerr << d->value_sequences[i].size()  << " ";
	}
	std::cerr << std::endl;
	for(size_t i = 0; i < d->value_label_sequences.size(); i++){
	  std::cerr << d->value_label_sequences[i].size()  << " ";
	}
	std::cerr << std::endl;*/
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
	    	
            //std::cerr << d->value_sequences[d->cursors[i].first][d->cursors[i].second + step] << " ";
            //std::cerr << d->value_label_sequences[d->cursors[i].first][d->cursors[i].second + step + 1]<< std::endl;

	    this->value_feat_cpu.data[i] = d->value_sequences[d->cursors[i].first][d->cursors[i].second + step];
            this->value_label_cpu.data[i] = d->value_sequences[d->cursors[i].first][d->cursors[i].second + step + 1];
        }
        
        feat.CopyFrom(this->value_feat_cpu);
        label.CopyFrom(this->value_label_cpu);
    }

 
    virtual void LoadTime(IDataLoader* d, IMatrix<mode, Dtype>* g_feat, IMatrix<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step) = 0;

    SparseMat<CPU, Dtype> event_feat_cpu, event_label_cpu; 
    DenseMat<CPU, Dtype> value_feat_cpu, value_label_cpu;    
};

template<MatMode mode>
class SingleTimeLoader : public IEventTimeLoader<mode>
{
public:
    SingleTimeLoader() : IEventTimeLoader<mode>() {}

    virtual void LoadTime(IDataLoader* d, IMatrix<mode, Dtype>* g_feat, IMatrix<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step) override
    {
        auto& feat = g_feat->DenseDerived();
        auto& label = g_label->DenseDerived();
        
        this->time_feat_cpu.Resize(cur_batch_size, 1);
        this->time_label_cpu.Resize(cur_batch_size, 1);
         
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            
	    this->time_feat_cpu.data[i] = d->time_sequences[d->cursors[i].first][d->cursors[i].second + step];
            this->time_label_cpu.data[i] = d->time_label_sequences[d->cursors[i].first][d->cursors[i].second + step];
        }
        
        feat.CopyFrom(this->time_feat_cpu);
        label.CopyFrom(this->time_label_cpu);
    }

    DenseMat<CPU, Dtype> time_feat_cpu, time_label_cpu;
};

template<MatMode mode>
class UnixTimeLoader : public IEventTimeLoader<mode>
{
public:
    UnixTimeLoader() : IEventTimeLoader<mode>() {}

    virtual void LoadTime(IDataLoader* d, IMatrix<mode, Dtype>* g_feat, IMatrix<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step) override
    {
        auto& feat = g_feat->SparseDerived();
        auto& label = g_label->DenseDerived();

        time_feat_cpu.Resize(cur_batch_size, cfg::time_dim);
        time_label_cpu.Resize(cur_batch_size, 1);

        time_feat_cpu.ResizeSp(cfg::unix_str.size() * cur_batch_size, cur_batch_size + 1);

        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            time_t tt = (time_t)d->time_sequences[d->cursors[i].first][d->cursors[i].second + step];            
            struct tm *ptm = localtime(&tt);

            time_feat_cpu.data->ptr[i] = i * cfg::unix_str.size();

            int cur_dim = 0, col;
            for (size_t j = 0; j < cfg::unix_str.size(); ++j)
            {
                switch (cfg::unix_str[j])
                {
                    case 'y':
                        col = ptm->tm_year;
                        break;
                    case 'm':
                        col = ptm->tm_mon;
                        break;
                    case 'd':
                        col = ptm->tm_mday - 1;
                        break;
                    case 'w':
                        col = ptm->tm_wday;
                        break;
                    case 'H':
                        col = ptm->tm_hour;
                        break;
                    case 'M':
                        col = ptm->tm_min;
                        break;
                    default:                        
                        assert(false);
                        break;
                }
                time_feat_cpu.data->col_idx[i * cfg::unix_str.size() + j] = col + cur_dim;
                time_feat_cpu.data->val[i * cfg::unix_str.size() + j] = 1.0;
                cur_dim += cfg::field_dim[cfg::unix_str[j]]; 
            }

            time_label_cpu.data[i] = d->time_label_sequences[d->cursors[i].first][d->cursors[i].second + step];   
        }

        time_feat_cpu.data->ptr[cur_batch_size] = cfg::unix_str.size() * cur_batch_size;

        feat.CopyFrom(this->time_feat_cpu);
        label.CopyFrom(this->time_label_cpu);
    }

    DenseMat<CPU, Dtype> time_label_cpu;
    SparseMat<CPU, Dtype> time_feat_cpu;
};


DataLoader<TRAIN>* train_data;
DataLoader<TEST>* test_data, *val_data;


#endif
