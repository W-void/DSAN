import utils.CONST as cst
import itertools as it

test = "dense->cat_emb/norm"
inp = "input,tgt->function1/founction2/function3"


def parse_cmd(ori_inp):
	sep = "->"
	inp, func = ori_inp.split(sep)
	inps = inp.split(",")
	funcs = func.split("/")
	return inps, funcs


def tran_var(inps):
	return [eval(x) for x in inps]


def cmd_funcs(inp, func):
	return eval(func)(inp)


def pipe(data, *transforms):
	for t in transforms:
		data = t(data)
	return data


func1 = lambda x: sum(x) + 1
func2 = lambda x: x + 2
func3 = lambda x: x + 3

if __name__ == '__main__':
	input = 1
	tgt = 2
	cmd = "input,tgt->func1/func2/func2"

	inps, funcs = parse_cmd(cmd)
	inps = tran_var(inps)
	funcs = [eval(func) for func in funcs]
	print inps
	print func
	rs = pipe(inps, *funcs)
	print rs
