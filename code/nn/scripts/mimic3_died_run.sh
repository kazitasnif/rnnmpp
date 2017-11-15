#!/bin/bash

task=mimic3
prefix_event=event_d
prefix_time=time_d
prefix_value=value_d

DATA_ROOT=$HOME/rnnmpp/data/real/$task
RESULT_ROOT=$HOME/scratch/results/rnnmpp

hist=0
gru=0
n_embed=32
H=16
h2=16
bsize=512
bptt=10
learning_rate=0.0001
max_iter=140000
cur_iter=0
T=0
w_scale=0.01
mode=CPU
net=joint_value	
loss=intensity
lambda=1.32337245
#lambda=1
time_scale=1
save_dir=$RESULT_ROOT/$net-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/main \
    -h2 $h2 \
    -history $hist \
    -gru $gru \
    -t_scale $time_scale \
    -lambda $lambda \
    -loss $loss \
    -event $DATA_ROOT/$prefix_event \
    -time $DATA_ROOT/$prefix_time \
    -value $DATA_ROOT/$prefix_value \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -save_eval 0 \
    -save_test 1 \
    -l2 0.0 \
    -T $T \
    -m 0.9 \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 500 \
    -int_test 5000 \
    -int_save 25000 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
