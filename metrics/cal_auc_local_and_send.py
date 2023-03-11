import pandas as pd
# from utils.utils import tick_tock
from utils.tools import tick_tock
from utils.parse_cmd import init_arguments
from utils.tools import approximate_auc_1, logloss
import os
import glob
import os
from send import *
import json

args = init_arguments()
exp_path = os.path.join(args.exp, "rs")
exp_conf = os.path.join(args.exp, "task_conf.json")


def load_json(config_file):
	params = {}
	data = json.load(open(config_file))
	for k, v in data.iteritems():
		params[k] = v
	return params


def is_non_zero_file(fpath):
	return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def read_one(f):
	flag = is_non_zero_file(f)
	if flag:
		rs = pd.read_csv(f, delimiter='\t')
		rs.columns = ['labels', 'pred']
		# print rs.head(10)
		return rs
	else:
		return None


with tick_tock("read data") as f:
	file_list = glob.glob(exp_path + "/part*")
	dfs = filter(lambda x: x is not None, map(read_one, file_list))
	conf = load_json(exp_conf)
	dfs = pd.concat(dfs, axis=0)
# print dfs.head(10)

with tick_tock("cal auc") as f:
	auc = approximate_auc_1(dfs.labels.tolist(), dfs.pred.tolist())
	logloss = logloss(dfs.labels.tolist(), dfs.pred.tolist())
	auc = "{} final auc is {},logloss is {}.".format(args.exp, auc, logloss)
	use_conf = "Input: {} \nMerge: {} \nClass: {} \nDate: train date from {} to {} test date {}".format("\n".join(conf['input_units']),
	                                                            " ".join(conf.get("merge_units", ["merge None"])),
	                                                            conf.get("model_class_name", "class None"), conf['train_data_start'], conf['train_data_end'], conf['test_data_start'])
	des = dfs.describe()

	print auc
	print use_conf
	print des

	send_dx_message(auc, [conf['user_name']], '137820335797')
	send_dx_message(use_conf, [conf['user_name']], '137820335797')
	send_dx_message(des.to_string(), [conf['user_name']], '137820335797')

	send_dx_group_message(auc, '66638842359', '137820335797')
	send_dx_group_message(use_conf, '66638842359', '137820335797')
	send_dx_group_message(des.to_string(), '66638842359', '137820335797')

# send_dx_group_message('hello world', '64013817124', '137550031793')
