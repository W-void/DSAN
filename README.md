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