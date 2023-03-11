#!/usr/bin/env bash
# set -x

source ~/.bashrc
source /opt/meituan/tensorflow-release/local_env.sh
source /opt/meituan/hadoop/bin/hadoop_user_login.sh hadoop-hmart-waimaiad

export user_group="hadoop-hmart-waimaiad"
export HOME=/home/sankuai
export HOPE_BA_CLIENT_ID="waimaiad"
export HOPE_BA_CLIENT_SECRET="QBkdQRtJhBqACyzL2lcUO5lmmngs16pQ"
export HOPE_BA_SERVICE="AifreeHopeBa"

export HADOOP_HOME=/opt/meituan/hadoop
export AFO_TF_HOME=/opt/meituan/tensorflow-release
export HADOOP_USER=hadoop-hmart-waimaiad
source /opt/meituan/hadoop/bin/hadoop_user_login.sh ${HADOOP_USER}
source ${AFO_TF_HOME}/local_env.sh
mpi_submit=/opt/meituan/tensorflow-release/bin/mpi-submit.sh

function get_conf_from_json() {
    json_file=$1
    key=$2
    value=`cat ${json_file} | grep \"${key}\": | sed "s/\"${key}\": //g" | sed "s/[ \",\n\t]//g"`
    echo ${value}
}

afo_base_image_name="registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_afo-base-julang-test-dd7302b3"
afo_docker_image_name="registryonline-hulk.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/data-hadoop-hdp_tf1.15_mt_julang_1.0.0_cuda11.0_runtime-f5b15357"
exp=$1

config_file="task_conf.json"
data_struct_file="data_struct.json"

model_name=${exp##*\/}
user_name=`get_conf_from_json ${exp}/${config_file} user_name`
export HOPE_BA_USER=${user_name}

train_data_path=`get_conf_from_json ${exp}/${config_file} train_data_path`
test_data_path=`get_conf_from_json ${exp}/${config_file} test_data_path`
train_data_start=`get_conf_from_json ${exp}/${config_file} train_data_start`
train_data_end=`get_conf_from_json ${exp}/${config_file} train_data_end`
test_data_start=`get_conf_from_json ${exp}/${config_file} test_data_start`
test_data_end=`get_conf_from_json ${exp}/${config_file} test_data_end`

train_data=`hadoop fs -du -h ${train_data_path} | awk -F" " '{if($1>0) {print $NF}}' | awk -F"/" -v dt1=${train_data_start} -v dt2=${train_data_end} '{if($NF>=dt1 && $NF<=dt2) {print $0}}' | awk '{printf "%s,", $0}'`
test_data=`hadoop fs -du -h ${test_data_path} | awk -F" " '{if($1>0) {print $NF}}' | awk -F"/" -v dt1=${test_data_start} -v dt2=${test_data_end} '{if($NF>=dt1 && $NF<=dt2) {print $0}}' | awk '{printf "%s,", $0}'`

model_path=`get_conf_from_json ${exp}/${config_file} model_path`
model_dir=${model_path}/${model_name}/${train_data_start}_${train_data_end}

task=`get_conf_from_json ${exp}/${config_file} task`
if [[ $2 != '' ]];then
    task=$2
fi

if [[ ${task} == "train" ]];then
    dist_xml_name="dist_train_k8s.xml"
    hadoop fs -rm -r ${model_dir}
elif [[ ${task} == "evaluate" ]];then
    dist_xml_name="dist_eval_k8s.xml"
elif [[ ${task} == "save" ]];then
    dist_xml_name="dist_save_k8s.xml"
else
    echo "Task type error!!!"
    exit 1
fi

if [[ ${task} == "train" ]];then
    input_data=${train_data%?}
else
    input_data=${test_data%?}
fi

if [[ ${task} == "evaluate" ]];then
    echo "rm rs before:" ${model_dir}/rs
    hrmr ${model_dir}/rs
else
    echo "pass"
fi

echo "Submit info: exp:${exp}, task:${task}, afo_xml:${dist_xml_name}, model_name:${model_name}"
echo "Input data: ${input_data}"
echo "Model path: ${model_dir}"


${AFO_TF_HOME}/bin/mpi-submit.sh -conf ${exp}/${dist_xml_name} \
        --files gpu_train_eval.py,data,handle_layer,model,utils \
        --files ${exp}/${config_file},${exp}/${data_struct_file} \
        -Dafo.xm.notice.receivers.account=${user_name} \
        -Dargs.use_cog=True \
        -Dboard.log_dir=${model_dir} \
        -Dafo.afo-base.image.name=${afo_base_image_name} \
        -Dafo.docker.image.name=${afo_docker_image_name} \
        -Dargs.task=${task} \
        -Dargs.train_data=${input_data} \
        -Dargs.config_file=${config_file} \
        -Dargs.data_struct_file=${data_struct_file} \
        -Dargs.model_dir=${model_dir} \
        -Dargs.model_ckpt_dir=${model_dir} \
        -Dafo.app.name=${task}_${model_name} \
        -Dargs.model_ckpt_dir=${model_dir} \
        -Dargs.pass_size=${pass_size} \
        -Dargs.config_file=${config_file} \
        -Dafo.data.agent.memory=20480 \
        -Djulang.args.param_shm_size="1000G" \
        -Djulang.args.shard_num=8 \
        -Dafo.role.worker.env.LD_PRELOAD="/lib64/libtcmalloc.so" \
        -Dafo.tensorflow.cog.mode=2 \
        -Dafo.data.random.seed="3" \
        -Dafo.data.dispatch.policy="com.meituan.hadoop.afo.tensorflow.data.policy.FixedOrderPolicy" \
        -Dafo.data.fixed.order.worker.shuffle=true