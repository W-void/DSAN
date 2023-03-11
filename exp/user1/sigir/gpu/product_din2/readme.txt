
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/product_din2 train &
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/product_din2 evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/xxx/sigir/gpu/product_din2 &
