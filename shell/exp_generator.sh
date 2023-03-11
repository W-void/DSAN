#!/usr/bin/env bash

# 先手动创建目录, 目录下至少创建: data_struct.json 和 task_conf.json, task_conf.json 不包含 input_units

exp_path=$1
zero_name=$2
# exp_path="exp/zhangjin11/tmp_exp_path"
# zero_name="zero_exp"

if [[ ! -d ${exp_path}/${zero_name} ]];then
    echo "${exp_path}/${zero_name} not exist"
    cp -r exp/zhangjin11/exp_no_input_units ${exp_path}/${zero_name}
fi

# module_list=("handle_cat_dense_unit:Dense" "handle_cat_dense_unit:Category" "handle_cat_dense_unit:RerankDense" \
#             "handle_cat_dense_unit:LASTDense" "handle_cat_dense_unit:PoiTextEmbedDense" "handle_cat_dense_unit:QueryFeature" \
#             "handle_user_rerank_unit:CategoryRerank" "handle_chain_unit:ChainCacheFeature" "handle_user_path_unit:DecisionPathV1")
module_list=("handle_cat_dense_unit:Dense" "handle_cat_dense_unit:Category")

exp_name=""
module_list_str=""
for module in ${module_list[@]};do
    exp_name="${exp_name}_${module#*:}"
    exp_name=${exp_name#_}
    if [[ ! (${exp_name} == *Dense* && ${exp_name} == *Category*) ]];then
        echo "input module must contain Dense and Category"
        continue
    fi

    task_path=${exp_path}/${exp_name}
    echo ${task_path}
    if [[ -d ${task_path} ]];then
        echo "before task remove ${task_path}"
        rm -rf ${task_path}
     fi
    cp -r ${exp_path}/${zero_name} ${task_path}

    module_list_str="${module_list_str},\"${module}\""
    module_list_str=${module_list_str#,}

    input_units="\"input_units\": [${module_list_str}],"
    echo ${input_units}

    head -1 ${task_path}/task_conf.json >> ${task_path}/task_conf.json.tmp
    echo "    "${input_units} >> ${task_path}/task_conf.json.tmp
    echo >> ${task_path}/task_conf.json.tmp
    tail -n +2 ${task_path}/task_conf.json >> ${task_path}/task_conf.json.tmp

    rm ${task_path}/task_conf.json
    mv ${task_path}/task_conf.json.tmp ${task_path}/task_conf.json

    nohup sh shell/total_chain__submit_train_eval_calauc.sh ${task_path} &
    echo

    sleep 10
done
