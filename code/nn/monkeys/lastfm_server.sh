#!/bin/bash

task=lastfm
prefix_event=event_split_1000
prefix_time=time_split_1000

DATA_ROOT=$HOME/data/Research/NeuralPointProcess/data/real/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

n_embed=$1
H=$2
bsize=$3
bptt=$4
w_scale=$5
learning_rate=0.01
max_iter=4000
cur_iter=0
T=0
mode=CPU
net=joint
loss=mse
lambda=0.690150253058
save_dir=$RESULT_ROOT/$net-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize-wscale-$w_scale

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

../build/main \
    -event $DATA_ROOT/$prefix_event \
    -time $DATA_ROOT/$prefix_time \
    -loss $loss \
    -lambda $lambda \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -save_eval 0 \
    -save_test 1 \
    -T $T \
    -m 0.9 \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 100 \
    -int_test 2500 \
    -int_save 2500 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
