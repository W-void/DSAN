
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/merge_all train &
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/merge_all evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/xxx/sigir/gpu/merge_all &
