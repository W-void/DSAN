exp/user/base/
nohup sh shell/submit_train_eval.sh exp/user/base &


######### CPU

sh shell/submit_dist_task_speed.sh $path train >> rs.log 2>&1

sh shell/download_data.sh exp/user/dongjian/fasteval_newdense/
sh shell/cal_auc.sh exp/user/dongjian/fasteval_newdense/

sh shell/submit_dist_task_speed.sh exp/dongjian/fasteval/fasteval_newdense train
sh shell/total_chain__submit_train_eval_calauc.sh exp/user/dongjian/fasteval/fasteval_rmnewdense &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_rmnewdense &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_rmpoi &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_newdense &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_newdense_42day &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/fasteval/fasteval_newdense_42day_2 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/fasteval_newdensev2 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/fasteval_newdensev2_merge_zhangjin &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/fasteval_newdensev2_merge_zhangjin/fasteval_newdensev2_merge_zhangjin2 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_test &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_all &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/fasteval_newdensev2_merge_zhangjin/fasteval_newdensev2_merge_zhangjin3 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_all1 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_att1 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_pool1 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/cat_search_all2 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/search_input/ccat_search_all &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/fasteval_newcat2 &

nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/fasteval_new_base_20230110 &
nohup sh shell/total_chain__eval_calauc.sh exp/team/2023H1/search_module_20230109/search_module_v1/search_module_v1_input_units_6_add_CategoryKws &


nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch1 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin2 &
nohup sh shell/total_chain__eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin2 &

########gpu


nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3_att_f >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3_pool_f >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3a >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3_att_fa >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__calauc.sh exp/dongjian/2023H1/M1/gpu_catsearch_v3 >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_v1_test >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_test_senes >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__eval_calauc.sh exp/dongjian/2023H1/M1/gpu_test_senes >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_test_senes_test_gather >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/team/2023H1/search_module_20230109/gpu_search_module/gpu_search_module_input_units_0_add_Category >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/team/2023H1/search_module_20230109/gpu_search_module/gpu_search_module_input_units_1_add_RerankDense >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_test_senes_test_gather >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__eval_calauc.sh exp/team/2023H1/search_module_20230109/gpu_search_module_v1/gpu_search_module_v1_input_units_9_add_CategoryRerank >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__eval_calauc.sh exp/team/2023H1/search_module_20230109/gpu_search_module_v1/gpu_search_module_v1_input_units_3_add_PoiTextEmbedDense >> rs.log 2>&1 &

2023年01月12日 星期四
#1. lr实验
#2. merge 实验
#3. 测试cat_search base
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_test_senes_test_gather/gpu_test_senes_test_gather_nond >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/adgpu_cat_search_v4_backr >> rs.log 2>&1 &

nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/adgpu_cats_v4_test1 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_decisionv2 >> rs.log 2>&1 &

2023年01月13日 星期五
# 1. lr 为啥越大越好？？？
# 2. merge: 构造更宽、多组交互。
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin1 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin_1m >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin2 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin_7day >> rs.log 2>&1 &

# 3. 看下各种fuxi ctr 啥能抄过来。
# 4. 之前的mlp结构。
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_nomerge >> rs.log 2>&1 &


2023年01月14日 星期六
#1. 特征、样本、模型。模型需要搜一波（mlp、lr、module）
#2. fux 不行，pytorch的。

2023年01月15日 星期日

catsearch cpu oom
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch1 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch/cpu_catsearch_samegpu >> rs.log 2>&1 &
nohup sh shell/total_chain__eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch &
nohup sh shell/total_chain__eval_calauc.sh exp/dongjian/2023H1/M1/cpu_catsearch1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/adgpu_cats_v4_test1 >> rs.log 2>&1 &



重训一下
mergedin  cpu oom
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin2 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin/cpu_merge_outdin_001_repeatgpu_outdin &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin/cpu_merge_outdin_002_repeatgpu_tgtdin &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin/cpu_merge_outdin_003_repeatgpu_outdin &


nohup sh shell/total_chain__eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin2 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin3 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin/cpu_merge_outdin3 &
nohup sh shell/total_chain__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/cpu_merge_outdin/cpu_merge_outdin4 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin_001_8wei >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin_002_only_outdin >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_merge_lib/adgpu_merge_outdin_003_tgtdin >> rs.log 2>&1 &


attention 模块
新旧代码， 20% 采样，shuwei对齐。

2023年01月17日 星期二
修复decisoin v1
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_test_decisionv1fix_1>> rs.log 2>&1 &

测试单个attention效果
sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/gpu_nsplits/search_cmds.sh

-测试699
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/gpu_nsplits/gpu_nsplits_077_add699 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/adgpu_cat_search_v4/gpu_nsplits/gpu_nsplits_078_add699_nofix >> rs.log 2>&1 &


对齐最新日期gpu效果
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_newdate_base >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_003_newdate_base/gpu_001_newdate_addcatesearch >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_003_newdate_base/gpu_001_newdate_addcatesearch_v2 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_003_newdate_base/gpu_001_newdate_addcatesearch_v3 >> rs.log 2>&1 &


学习res
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_res_001_restest >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_res_001_restest/gpu_res_002_restestv2 >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_res_001_restest/gpu_res_003_restest >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_res_001_restest/gpu_res_004_1dcnn >> rs.log 2>&1 &


2023年01月19日 星期四
后续实验：
1. 搭建下docker调试环境。
2. attention deepctr类的跑一波。
3. paperwithcode attention。
4. autoattention.这个破玩意没啥用。
5. idcnn啥的 keras
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn/gpu_006_1dcnn_001_testnoshape >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn/gpu_006_1dcnn_002_testnoshape >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn/gpu_006_1dcnn_003_allfea >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn/gpu_006_1dcnn_004_keras >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_006_1dcnn/gpu_006_1dcnn_005_kerasv2 >> rs.log 2>&1 &

6. 学习autogluon

2023年01月23日 星期一
改造支持模块+输出。自定义。
-- 复用 clas-paras。通过装饰器实现。
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_002_class_paras/gpu_002_test_paras >> rs.log 2>&1 &

测试attr 与poi attention
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_004_multiattention >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_004_multiattention/gpu_004_multiattention_001 >> rs.log 2>&1 &

2023年01月31日 星期二
测试多hashtable和attention 的影响。
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_005_attsearch_newhashtable >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_005_attsearch_newhashtable/gpu_005_attsearch_newhashtable_001_findbase >> rs.log 2>&1 &
nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh exp/dongjian/2023H1/M1/gpu_005_attsearch_newhashtable/gpu_005_attsearch_newhashtable_002_byname >> rs.log 2>&1 &

