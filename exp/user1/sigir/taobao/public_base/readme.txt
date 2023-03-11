
nohup sh shell/gpu_submit_dist_task_speed_hope.sh exp/xxx/sigir/taobao/public_base train &
nohup sh shell/gpu_submit_dist_task_speed_hope.sh exp/xxx/sigir/taobao/public_base evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/xxx/sigir/taobao/public_base &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/xxx/sigir/taobao/public_base &