
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/clk_din train &
nohup sh shell/gpu_submit_dist_task_speed.sh exp/xxx/sigir/gpu/clk_din evaluate &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/xxx/sigir/gpu/clk_din &
