from functools import *


def test(func):
	print func.__name__
	return func


@test
def f(inp):
	print inp


f("a")


class base:
	def __init__(self):
		self.cat_gather_feat = []
		self.dense_gather_feat = []

	def gather_cat(self, feat_list):
		self.cat_gather_feat.append(feat_list)
		return feat_list


class a(base):

	def __init__(self):
		self.cat_feat = ["a", "b"]
		self.dense_feat = [1, 2]
