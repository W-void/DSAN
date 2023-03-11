import pandas as pd
# from utils.utils import tick_tock
from utils.tools import tick_tock, merge
from utils.parse_cmd import init_arguments
from utils.tools import approximate_auc_1, logloss
import os
import glob
import os
import json
import itertools as it
import utils.tools as tools
from handle_layer.feature_ana import all_feas_names
from collections import OrderedDict
import arrow as ar
import arrow

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
SHELL_CMDS = lambda name: ["nohup sh {}/{} >> rs.log 2>&1 &".format(name, x) for x in ["search_cmds.sh", "sleep_cmds.sh"]]

SLEEP_HOUR = 8
SLEEP_SECONEDS = SLEEP_HOUR * 3600
PARALLEL_ROUNDS = 7


def load_json(config_file):
	# type: (object) -> object
	params = json.load(open(config_file), object_pairs_hook=OrderedDict)
	return params


def is_non_zero_file(fpath):
	return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def generate_paras(rrs, do_product=False):
	# rrs = [func() for func in funcs]
	print rrs, "rrs"
	# rrs = list(it.chain(*[func() for func in funcs]))
	if do_product:
		rrs = list(it.product(*rrs))
		return [tools.merge(*rr) for rr in rrs]
	else:
		return list(it.chain(*rrs))


def lr_test(lr):
	tgt = "learning_rate"
	# lr = [50.0, 20.0, 10.0, 5.0, 1.0, 0.8, 0.5]
	return [{tgt: l} for l in lr]


def catsearch(key, names):
	method = "map_attention"
	tgt = key.split("#")
	return [{tgt[0]: {tgt[1]: "#".join([fea, method])}} for fea in names]


def module_seq_add(tgt_conf):
	base = ["handle_cat_dense_unit:Dense,Category"]

	def parse_conf(one_conf):
		def one_line(cc):
			_py = cc.split(":")[0]
			clss = cc.split(":")[1].split(",")
			rs = [":".join([_py, cls]) for cls in clss]
			return rs

		rs = list(it.chain(*[one_line(c) for c in one_conf]))
		return rs

	base_list = parse_conf(base)
	input_units_list = parse_conf(tgt_conf[modoule_target])
	diff = tools.diff(input_units_list, base_list)

	exps = []
	exps.append(base_list)
	tmp = []
	for d in diff:
		tmp.append(d)
		exps.append(base_list + tmp)
	return [{"input_units": x} for x in exps]


def date_gene(conf, ascending=False):
	train_start = ar.get(str(conf['train_data_start']))
	train_end = ar.get(str(conf['train_data_end']))

	print "train_data_end ???", train_end, conf['train_data_end']
	if ascending:
		rs = [{"train_data_start": train_end.shift(weeks=-i).format("YYYYMMDD"), "train_data_end": train_end.format(("YYYYMMDD"))} for i in range(1, 12) if
		      train_end.shift(weeks=-i) >= train_start]
		print "date_gene", rs
		return rs
	else:
		rs = [{"train_data_start": train_start.format("YYYYMMDD"), "train_data_end": train_start.shift(weeks=i).format("YYYYMMDD")} for i in range(1, 12) if train_start.shift(weeks=i) <= train_end]
		print "date_gene", rs
		return rs


def search_classparas():
	b = [["ScenesSeqAndFeedRec", 6], ["CategoryRerank", 6], ["DecisionPathV1Fix", 4], ["ChainCacheFeature", 5]]
	bn = [x[1] for x in b]
	gene_tup = lambda inp, len: {inp: ",".join(map(str, range(0, len + 1)))}
	rs = []
	for i in range(sum(bn)):
		if i < bn[0]:
			rs.append(gene_tup(b[0][0], i))

		elif i < sum(bn[:2]):
			i = i - sum(bn[:1])
			rs.append(merge(gene_tup(b[0][0], bn[0] - 1), gene_tup(b[1][0], i)))

		elif i < sum(bn[:3]):
			i = i - sum(bn[:2])
			rs.append(merge(gene_tup(b[0][0], bn[0] - 1), gene_tup(b[1][0], bn[1] - 1), gene_tup(b[2][0], i)))

		elif i < sum(bn):
			i = i - sum(bn[:3])
			rs.append(merge(gene_tup(b[0][0], bn[0] - 1), gene_tup(b[1][0], bn[1] - 1), gene_tup(b[2][0], bn[2] - 1), gene_tup(b[3][0], i)))
	rs = [{"class_paras": x} for x in rs]
	return rs


def write_confs(conf, paras):
	print "write_confs paras", paras

	def one_conf(conf, num, para):
		print "one para", para

		def one_para(p):
			if isinstance(para[p], dict):
				print "paras_name", ["_".join([k, v]) for k, v in para[p].items()]
				return "_".join(["_".join([k, v]) for k, v in para[p].items()])
			elif isinstance(para[p], list):
				tgt = para[p][-1]
				if ":" in tgt:
					return tgt.split(":")[-1]
			else:
				return "_".join([p, str(para[p])])

		paras_name = "_".join(map(one_para, para)).replace("#", "_").replace(",", "_")
		conf = conf.copy()
		conf[search_name] = "_".join([exp_name, str(num).rjust(3, "0"), str(paras_name)])
		for i in para:
			if isinstance(conf[i], dict):
				conf[i] = conf[i].copy()
				conf[i].update(para[i])
			else:
				conf[i] = para[i]
		print "dense layer gene {}".format(para)
		print "final conf {}".format(conf)
		return conf

	rs = [one_conf(conf, n, p) for n, p in enumerate(paras)]
	print "write_confs rs {}".format(rs)
	return rs


with tick_tock("read conf") as f:
	conf = load_json(exp_conf)
	dist_eval_k8s_content = open(dist_eval_k8s, 'r').readlines()
	dist_save_k8s_content = open(dist_save_k8s, 'r').readlines()
	dist_train_k8s_content = open(dist_train_k8s, 'r').readlines()
	data_struct_content = open(data_struct, 'r').readlines()

with tick_tock("search gene") as f:
	base = ["handle_cat_dense_unit:Dense,Category"]
	conf['base'] = base
	# paras = generate_paras([catsearch("class_paras#CateSearchByName", all_feas_names[:5]), lr_test([1.0, 2.0])], do_product=False)
	# paras = generate_paras([search_classparas()], do_product=False)
	# paras = generate_paras([module_seq_add(conf)], do_product=False)
	paras = generate_paras([date_gene(conf, ascending=False)], do_product=False)

	confs = write_confs(conf, paras)

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
			print "write conf ", conf
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
			if i > 0 and i % PARALLEL_ROUNDS == 0:
				f_rs.append("sleep {}".format(SLEEP_SECONEDS))
			f_rs.append(r)
		w.writelines("\n".join(f_rs))

	with open(os.path.join(cur_path, args.exp, "search_cmds.sh"), 'w') as w:
		w.writelines("\n".join(rs))

	with open(os.path.join(cur_path, args.exp, "shell_cmds"), 'w') as w:
		shell_cmds = SHELL_CMDS(os.path.join(args.exp))
		w.writelines("\n".join(shell_cmds))
