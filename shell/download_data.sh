#!/usr/bin/env bash
set -x

source ~/.bashrc
source /opt/meituan/tensorflow-release/local_env.sh
source /opt/meituan/hadoop/bin/hadoop_user_login.sh hadoop-hmart-waimaiad

export user_group="hadoop-hmart-waimaiad"
export HOME=/home/sankuai
export HOPE_BA_CLIENT_ID="waimaiad"
export HOPE_BA_CLIENT_SECRET="QBkdQRtJhBqACyzL2lcUO5lmmngs16pQ"
export HOPE_BA_SERVICE="AifreeHopeBa"

hope_submit=/home/sankuai/.local/bin/hope

function get_conf_from_json() {
    json_file=$1
    key=$2
    value=`cat ${json_file} | grep \"${key}\": | sed "s/\"${key}\": //g" | sed "s/[ \",\n\t]//g"`
    echo ${value}
}


exp=$1
file_name=$2

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

model_path=`get_conf_from_json ${exp}/${config_file} model_path`
model_dir=${model_path}/${model_name}/${train_data_start}_${train_data_end}

task=`get_conf_from_json ${exp}/${config_file} task`
if [[ $2 != '' ]];then
    task=$2
fi


echo "Submit info: exp:${exp}, task:${task}, afo_xml:${dist_xml_name}, model_name:${model_name}"
echo "Input data: ${input_data}"
echo "Model path: ${model_dir}"

mkdir tmp
hget ${model_dir}/rs ${exp}"/rs"

echo "data write in " ${exp}"/rs"


