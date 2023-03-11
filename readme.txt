注意：所有执行脚本的目录为项目根目录，不要去shell底下执行会报错！！！
注意：注意文件夹名称就是model_name！！！最后没有斜杠！！！
注意：最新cpu base 为exp/dongjian/fasteval/fasteval_newdense

#### CPU 流程

0. 创建自己的实验路径，将exp/user/base下的文件拷备到实验路径下

1. 训练Train
   sh shell/submit_dist_task_speed.sh [实验路径，如exp/dongjian/fasteval] train

2. 评估Test
   sh shell/submit_dist_task_speed.sh [实验路径，如exp/dongjian/fasteval] evaluate


3. 计算auc。使用多worker评估。
    sh shell/download_data.sh exp/dongjian/fasteval/
    sh shell/cal_auc.sh exp/dongjian/fasteval/
    对应eval.xml 也进行了修改。最终结果在exp/user/dongjian/fasteval/final_auc，可以commit上来。

4. 大家可以使用total_chain__submit_train_eval_calauc来全流程得到结果。
    nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_newdense &


#### GPU 流程
0. 创建自己的实验路径，将exp/dongjian/2023H1/M1/adgpu_cat_search_v4 下的文件拷备到实验路径下. 注意复制gpu下的实验的东西，改了很多xml配置。

1. 训练Train
   sh shell/gpu_submit_dist_task_speed.sh [实验路径，如exp/dongjian/2023H1/M1/adgpu_cat_search_v4] train

2. 评估Test
   sh shell/gpu_submit_dist_task_speed.sh [实验路径，如exp/dongjian/2023H1/M1/adgpu_cat_search_v4] evaluate

3. 计算auc。使用多worker评估。
    sh shell/download_data.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4
    sh shell/cal_auc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4
    对应eval.xml 也进行了修改。最终结果在exp/dongjian/2023H1/M1/adgpu_cat_search_v4/final_auc，可以commit上来。

4. 大家可以使用total_chain_gpu__submit_train_eval_calauc来全流程得到结果。
    nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4


#### 单机调试
1. 创建docker

2. 从hdfs拷贝数据tfrecord文件，保存到单机下的路径 data_dir，比如我这里的目录是：
    /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/wangdongdong26/dataset/tf_records/new_dianjin/20221201
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
