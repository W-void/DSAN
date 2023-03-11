import pandas as pd
# from utils.utils import tick_tock
from utils.tools import tick_tock
from utils.parse_cmd import init_arguments
from utils.tools import approximate_auc_1, logloss
import os
import glob
import os
import json
import itertools as it
import utils.tools as tools

args = init_arguments()
utype = args.utype
exp_name = args.exp.split("/")[-1]
exp_path = os.path.join(args.exp, "rs")
exp_conf = os.path.join(args.exp, "task_conf.json")
data_struct = os.path.join(args.exp, "data_struct.json")
dist_eval_k8s = os.path.join(args.exp, "dist_eval_k8s.xml")
dist_save_k8s = os.path.join(args.exp, "dist_save_k8s.xml")
dist_train_k8s = os.path.join(args.exp, "dist_train_k8s.xml")
cur_path = os.path.abspath(os.getcwd())

CMD_CPU_TOTAL = "total_chain__submit_train_eval_calauc.sh"
CMD_GPU_TOTAL = "total_chain_gpu__submit_train_eval_calauc.sh"
modoule_target = "input_units"
search_name = "search_name"
CMD_GPU_TOTAL = lambda name: "nohup sh shell/total_chain_gpu__submit_train_eval_calauc.sh {} >> rs.log 2>&1 &".format(name)
CMD_CPU_TOTAL = lambda name: "nohup sh shell/total_chain__submit_train_eval_calauc.sh {} >> rs.log 2>&1 &".format(name)


def load_json(config_file):
	# type: (object) -> object
	params = {}
	data = json.load(open(config_file))
	for k, v in data.iteritems():
		params[k] = v
	return params


def is_non_zero_file(fpath):
	return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def module_add(conf, base, module_cands):
	def parse_conf(c):
		def one_line(cc):
			_py = cc.split(":")[0]
			clss = cc.split(":")[1].split(",")
			rs = [":".join([_py, clss]) for cls in clss]
			return rs

		rs = list(it.chain(*[one_line(c) for c in conf]))
		return rs

	target = "input_units"
	base = ['handle_cat_dense_unit:Dense,Category']

	base_list = parse_conf(base)
	input_units_list = parse_conf(conf[target])
	# mod_class_add =
	pass


def module_seq_add(conf):
	def parse_conf(one_conf):
		def one_line(cc):
			_py = cc.split(":")[0]
			clss = cc.split(":")[1].split(",")
			rs = [":".join([_py, cls]) for cls in clss]
			return rs

		rs = list(it.chain(*[one_line(c) for c in one_conf]))
		return rs

	def one_conf(conf, lr, num):
		conf = conf.copy()
		print lr
		print str(lr[-1].split(":")[-1])
		conf[search_name] = "_".join([exp_name, modoule_target, str(num).rjust(3, "0"), "add_" + str(lr[-1].split(":")[-1])])
		conf[modoule_target] = lr
		return conf

	base = conf['base']
	base_list = parse_conf(base)
	input_units_list = parse_conf(conf[modoule_target])
	diff = tools.diff(input_units_list, base_list)

	exps = []
	exps.append(base_list)
	tmp = []
	for d in diff:
		tmp.append(d)
		exps.append(base_list + tmp)
	print exps
	return [one_conf(conf, l, n) for n, l in enumerate(exps)]


def module_one_add(conf):
	def parse_conf(one_conf):
		def one_line(cc):
			_py = cc.split(":")[0]
			clss = cc.split(":")[1].split(",")
			rs = [":".join([_py, cls]) for cls in clss]
			return rs

		rs = list(it.chain(*[one_line(c) for c in one_conf]))
		return rs

	def one_conf(conf, lr, num):
		conf = conf.copy()
		print lr
		print str(lr[-1].split(":")[-1])
		conf[search_name] = "_".join([exp_name, modoule_target, str(num).rjust(3, "0"), "add_" + str(lr[-1].split(":")[-1])])
		conf[modoule_target] = lr
		return conf

	base = conf['base']
	base_list = parse_conf(base)
	input_units_list = parse_conf(conf[modoule_target])
	diff = tools.diff(input_units_list, base_list)

	exps = []
	exps.append(base_list)
	tmp = []
	for d in diff:
		# tmp.append(d)
		exps.append(base_list + [d])

	return [one_conf(conf, l, n) for n, l in enumerate(exps)]


def module_one_minus(conf):
	def parse_conf(one_conf):
		def one_line(cc):
			_py = cc.split(":")[0]
			clss = cc.split(":")[1].split(",")
			rs = [":".join([_py, cls]) for cls in clss]
			return rs

		rs = list(it.chain(*[one_line(c) for c in one_conf]))
		return rs

	def one_conf(conf, lr, num):
		conf = conf.copy()
		minus_tgt = lr
		conf[modoule_target] = tools.diff(input_units_list, [minus_tgt])
		print "input_units_list len is  {}".format(len(input_units_list))
		print "minus {} rs is {} len is {}".format(minus_tgt, conf[modoule_target], len(conf[modoule_target]))
		class_name = str(minus_tgt.split(":")[-1])
		conf[search_name] = "_".join([exp_name, modoule_target, str(num).rjust(3, "0"), "minus_" + class_name])

		return conf

	base = conf['base']
	base_list = parse_conf(base)
	input_units_list = parse_conf(conf[modoule_target])

	return [one_conf(conf, l, n) for n, l in enumerate(["all"] + input_units_list)]


def lr_search(conf):
	lr_test = [1.0, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
	lr_test = [1000, 500, 200, 100, 50, 10, 1]

	tgt = 'learning_rate'

	def one_conf(conf, lr, num):
		conf = conf.copy()
		conf[search_name] = "_".join([exp_name, tgt, str(num).rjust(3, "0"), str(lr)])
		conf[tgt] = lr
		return conf

	return [one_conf(conf, l, n) for n, l in enumerate(lr_test)]


def search_dense_layer(conf):
	dense_fir = [5000, 4000, 3000, 2048, 1500, 1024, 512, 256, 128, 64]
	dense_all = [[x, x / 2, x / 4] for x in dense_fir]
	tgt = 'dense_layer'

	def one_conf(conf, paras, num):
		paras_name = "_".join(map(str, paras))
		conf = conf.copy()
		conf[search_name] = "_".join([exp_name, tgt, str(num).rjust(3, "0"), str(paras_name)])
		conf[tgt] = paras
		print "dense layer gene {}".format(paras)
		return conf

	return [one_conf(conf, l, n) for n, l in enumerate(dense_all)]


def search_dense_lr(conf):
	dense_fir = [5000, 3000, 2048, 1024, 512, 256]
	dense_all = [[x, x / 2, x / 4] for x in dense_fir]

	lr_test = [200, 100, 50, 10, 1, 1.0, 0.8, 0.5, 0.3, 0.1]

	exps = it.product(dense_all, lr_test)

	def search_dense_conf(conf, paras, num):
		dense_para = paras[0]
		lr_para = paras[1]

		tgt_dense = 'dense_layers'
		tgt_lr = 'learning_rate'

		dense_para_name = "_".join(map(str, dense_para))
		conf = conf.copy()
		conf[tgt_dense] = dense_para

		lr_para_name = lr_para
		conf = conf.copy()
		conf[tgt_lr] = lr_para

		conf[search_name] = "_".join([exp_name, str(num).rjust(3, "0"), str(dense_para_name), str(lr_para_name)])

		return conf

	return [search_dense_conf(conf, l, n) for n, l in enumerate(exps)]


def generate_confs(conf, funcs):
	return list(it.chain(*[func(conf) for func in funcs]))


def generate_cmds(conf, cmds):
	pass


def write_conf(conf, name):
	tgt_path = conf
	pass


with tick_tock("read conf") as f:
	conf = load_json(exp_conf)
	dist_eval_k8s_content = open(dist_eval_k8s, 'r').readlines()
	dist_save_k8s_content = open(dist_save_k8s, 'r').readlines()
	dist_train_k8s_content = open(dist_train_k8s, 'r').readlines()
	data_struct_content = open(data_struct, 'r').readlines()

with tick_tock("search gene") as f:
	conf['base'] = base

	# dict gene
	# write conf
	confs = generate_confs(conf, [search_dense_lr])
	print "final tgt conf {}", [c[modoule_target] for c in confs]
	print "final tgt conf {}", [c['learning_rate'] for c in confs]

with tick_tock("write_conf") as f:
	def write_one_conf(conf):
		conf_dir = os.path.join(cur_path, args.exp, conf[search_name])
		if not os.path.exists(conf_dir):
			os.mkdir(conf_dir)

		with open(os.path.join(conf_dir, "task_conf.json"), 'w') as tc, \
				open(os.path.join(conf_dir, "dist_train_k8s.xml"), 'w') as tr, \
				open(os.path.join(conf_dir, "dist_eval_k8s.xml"), 'w') as ev, \
				open(os.path.join(conf_dir, "dist_save_k8s.xml"), 'w') as sa, \
				open(os.path.join(conf_dir, "data_struct.json"), 'w') as ds:
			json.dump(conf, tc, indent=4)
			tr.writelines(dist_train_k8s_content)
			ev.writelines(dist_eval_k8s_content)
			sa.writelines(dist_save_k8s_content)
			ds.writelines(data_struct_content)


	def gene_cmd(conf):
		if utype == 'CPU':
			return CMD_CPU_TOTAL(os.path.join(args.exp, conf[search_name]))
		if utype == 'GPU':
			return CMD_GPU_TOTAL(os.path.join(args.exp, conf[search_name]))


	_ = map(write_one_conf, confs)
	rs = map(gene_cmd, confs)

	with open(os.path.join(cur_path, args.exp, "sleep_cmds.sh"), 'w') as w:
		f_rs = []
		for i, r in enumerate(rs):
			if i > 0 and i % 5 == 0:
				f_rs.append("sleep 21600")
			f_rs.append(r)
		w.writelines("\n".join(f_rs))

	with open(os.path.join(cur_path, args.exp, "search_cmds"), 'w') as w:
		w.writelines("\n".join(rs))
