核心代码在（分数据集）：

./handle_layer/handle_lib/multi_scene_lib/meituan/meituan_merge.py
./handle_layer/handle_lib/multi_scene_lib/eleme/eleme_merge.py
./handle_layer/handle_lib/multi_scene_lib/taobao/taobao_merge.py

---
注意：所有执行脚本的目录为项目根目录，不要去shell底下执行会报错！！！
注意：注意文件夹名称就是model_name！！！最后没有斜杠！！！

#### CPU 流程

0. 创建自己的实验路径，将exp/user/\下的文件拷备到实验路径下

1. 训练Train
   sh shell/submit_dist_task_speed.sh [实验路径，如exp/user1/sigir/cpu/base_cpu] train

2. 评估Test
   sh shell/submit_dist_task_speed.sh [实验路径，如exp/user1/sigir/cpu/base_cpu] evaluate


3. 计算auc。使用多worker评估。
    sh shell/download_data.sh [实验路径，如exp/user1/sigir/cpu/base_cpu]
    sh shell/cal_auc.sh [实验路径，如exp/user1/sigir/cpu/base_cpu]
    对应eval.xml 也进行了修改。最终结果在[实验路径，如exp/user1/sigir/cpu/base_cpu]/final_auc，可以commit上来。

4. 大家可以使用total_chain__submit_train_eval_calauc来全流程得到结果。
    nohup sh shell/total_chain__submit_train_eval_calauc.sh [实验路径，如exp/user1/sigir/cpu/base_cpu]/fasteval_newdense &


#### GPU 流程
0. 创建自己的实验路径，将[实验路径，如exp/user1/sigir/gpu/base] 下的文件拷备到实验路径下. 注意复制gpu下的实验的东西，改了很多xml配置。

1. 训练Train
   sh shell/gpu_submit_dist_task_speed.sh [实验路径，如exp/user1/sigir/gpu/base] train

2. 评估Test
   sh shell/gpu_submit_dist_task_speed.sh [实验路径，如exp/user1/sigir/gpu/base] evaluate

3. 计算auc。使用多worker评估。
    sh shell/download_data.sh [实验路径，如exp/user1/sigir/gpu/base]
    sh shell/cal_auc.sh [实验路径，如exp/user1/sigir/gpu/base]
    对应eval.xml 也进行了修改。最终结果在[实验路径，如exp/user1/sigir/gpu/base]/final_auc，可以commit上来。

4. 大家可以使用total_chain_gpu__submit_train_eval_calauc来全流程得到结果。
    nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh [实验路径，如exp/user1/sigir/gpu/base]


#### 单机调试
1. 创建docker

2. 从hdfs拷贝数据tfrecord文件，保存到单机下的路径 data_dir，比如我这里的目录是：
    /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/XXX/dataset/tf_records/20221201
    路径下有part文件，part-r-00000。

3. 修改debug的配置文件：
    exp/user/debug/task_conf.json
    修改两处：
    a). tfrecord的路径，"train_data_path": "/your/data_dir"
    b). 模型保存的路径，"model_path": "/your/model_path"

4. 运行单机
    sh shell/submit_dist_task_speed_debug.sh exp/user/debug train

5. 设置断点，在python代码中增加pdb，进行断点调试。


#### 参数搜索
python tools/exp_generator.py --utype GPU --exp exp/team/2023H1/search_module_20230109/gpu_search_module_v2_obo
支持 module_one_add（base+1 modoule）、module_seq_add（[base+1]+1 module）、lr_search(learning rate)
