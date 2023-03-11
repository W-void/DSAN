path=$1
sh shell/gpu_submit_dist_task_speed.sh $path train >> rs.log 2>&1
sh shell/gpu_submit_dist_task_speed.sh $path evaluate >> rs.log 2>&1