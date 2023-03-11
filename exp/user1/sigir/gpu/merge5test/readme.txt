
nohup sh shell/gpu_submit_dist_task_speed.sh exp/wangshuli03/sigir/gpu/merge5test train &
nohup sh shell/gpu_submit_dist_task_speed.sh exp/wangshuli03/sigir/gpu/merge5 evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/wangshuli03/sigir/gpu/merge5 &
