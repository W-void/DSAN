path=$1
sh shell/submit_dist_task_speed.sh $path evaluate >> $path/rs.log 2>&1

sh shell/download_data.sh $path
sh shell/cal_auc.sh $path