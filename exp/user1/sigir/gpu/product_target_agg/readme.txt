
nohup sh shell/gpu_submit_dist_task_speed.sh exp/wangshuli03/sigir/gpu/product_target_agg train &
nohup sh shell/gpu_submit_dist_task_speed.sh exp/wangshuli03/sigir/gpu/product_target_agg evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/wangshuli03/sigir/gpu/product_target_agg &
