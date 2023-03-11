path=$1
sh shell/submit_dist_task_speed.sh $path train >> $path/rs.log 2>&1
if [ $? -ne 0 ];then
    echo "train model task  failed"
    exit -1
fi
sh shell/submit_dist_task_speed.sh $path evaluate >> $path/rs.log 2>&1

sh shell/download_data.sh $path
sh shell/cal_auc.sh $path