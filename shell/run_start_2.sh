#!/usr/bin/env bash
set -x
##注意：所有执行脚本的目录为项目根目录，不要去shell底下执行会报错！！！
##创建自己的实验路径，将exp/user/base下的文件拷备到实验路径下

log_pre="scenes_feedrec_seq0103"
exp_path="exp/user/scenes_feedrec_seq"

##1. 训练Train
##sh shell/submit_dist_task_speed.sh [实验路径，如exp/user/base] train
# :<<TRAIN
echo "# Train>> `date`"
nohup sh shell/submit_dist_task_speed.sh ${exp_path} train >logs/${log_pre}_train 2>&1 &
wait
echo "wait!!"
if [ $? -ne 0 ];then
    echo "train model task  failed"
    exit -1
fi
echo "Train SUCC!!! `date`"
#exit 0
TRAIN


##2. 评估Test
##   sh shell/submit_dist_task_speed.sh [实验路径，如exp/user/base] evaluate 
# eval
echo "# Test>> `date`"
nohup sh shell/submit_dist_task_speed.sh ${exp_path}  evaluate >logs/${log_pre}_eval 2>&1 &
wait
if [ $? -ne 0 ];then
    echo "test model task  failed"
    exit -1
fi
#exit 0 
#TRAIN


##3. 计算auc。使用多worker评估。
## sh shell/download_data.sh exp/user/dongjian/fasteval/
## sh shell/cal_auc.sh exp/user/dongjian/fasteval/
##（对应eval.xml 也进行了修改。）
#ZYP
echo "# hget&AUC >> `date`"
nohup sh shell/download_data.sh ${exp_path} ${log_pre}  >>logs/${log_pre}_eval 2>&1 &
wait
if [ $? -ne 0 ];then
    echo "test model task  failed"
    exit -1
fi
#exit 0 

nohup sh shell/cal_auc.sh ${exp_path} ${log_pre} >>logs/${log_pre}_eval 2>&1 &

echo "END cal AUC"
