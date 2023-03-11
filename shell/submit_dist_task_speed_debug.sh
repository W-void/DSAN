#!/usr/bin/env bash
# set -x
set +v
export user_group="hadoop-hmart-waimaiad"
export HOME=/home/sankuai
export HOPE_BA_CLIENT_ID="waimaiad"
export HOPE_BA_CLIENT_SECRET="QBkdQRtJhBqACyzL2lcUO5lmmngs16pQ"
export HOPE_BA_SERVICE="AifreeHopeBa"
function get_conf_from_json() {
    json_file=$1
    key=$2
    value=`cat ${json_file} | grep \"${key}\": | sed "s/\"${key}\": //g" | sed "s/[ \",\n\t]//g"`
    echo ${value}
}
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
#train_data=`hadoop fs -du -h ${train_data_path} | awk -F" " '{if($1>0) {print $NF}}' | awk -F"/" -v dt1=${train_data_start} -v dt2=${train_data_end} '{if($NF>=dt1 && $NF<=dt2) {print $0}}' | awk '{printf "%s,", $0}'`
#test_data=`hadoop fs -du -h ${test_data_path} | awk -F" " '{if($1>0) {print $NF}}' | awk -F"/" -v dt1=${test_data_start} -v dt2=${test_data_end} '{if($NF>=dt1 && $NF<=dt2) {print $0}}' | awk '{printf "%s,", $0}'`
time_prefix=`date "+%Y%m%d-%H%M"`
model_path=`get_conf_from_json ${exp}/${config_file} model_path`
model_dir=${model_path}/${model_name}/${train_data_start}_${train_data_end}
mkdir -p ${model_dir}
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
echo "Submit info: exp:${exp}, task:${task}, afo_xml:${dist_xml_name}, model_name:${model_name}"
echo "Model path: ${model_dir}"
python train_eval.py -mpdb \
              --task="train" \
              --train_data=${train_data_path} \
              --config_file=${exp}/${config_file} \
              --data_struct_file=${exp}/${data_struct_file} \
              --model_dir=${model_dir} \
              --model_ckpt_dir=${model_dir}
