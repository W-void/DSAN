#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import tensorflow as tf
from data.data_utils import index_of_tensor
import utils.CONST as cst


class GatherFea(object):
	def __init__(self):
		self.dense = []
		self.cat = []

	def append_dense_feas(self, feas_list):
		if isinstance(feas_list, list):
			self.dense = self.dense + feas_list
		else:
			raise Exception("fea_list must be list")

	def append_cat_feas(self, feas_list):
		if isinstance(feas_list, list):
			self.cat = feas_list + self.cat
		else:
			raise Exception("fea_list must be list")


class InputBase(object):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(InputBase, self).__init__()
		self.data_struct = data_struct
		self.params = params
		self.is_training = is_training
		self.logger = params['logger']
		self.hashtable = hashtable
		self.ptable_lookup = ptable_lookup  # ptable_lookup take list_ids,v_name

		self.dense_columns_info = self.data_struct.columns_dict['dense_feature']
		self.cat_columns_info = self.data_struct.columns_dict['cat_feature']
		self.aux_columns_info = self.data_struct.columns_dict['auxiliary_info']
		self.gather_feas = GatherFea()
		self.cat_list = []
		self.dense_list = []

		self.base_hashtable = self.params['base_hashtable']
		self.attention_switch_hub = self.params['attention_switch_hub']

		self.base_embed_dim = 8
		self.r = math.sqrt(6 / self.base_embed_dim)
		self.ALL = "ALL"
		self.NONE = "None"

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

	def eval_str_result(self, inp_list):
		class_paras = self.params.get(cst.class_paras, None)
		paras = class_paras.get(self.__class__.__name__) if class_paras is not None else None
		if paras == self.ALL or paras is None:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return inp_list

		if paras == self.NONE:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return None

		if paras or paras != self.ALL:
			self.out_str = paras.split(",")
			self.logger.info("eval_result work. paras is {}".format(self.out_str))
			return [eval("self." + x) for x in self.out_str]

	def eval_list_result(self, inp_list):
		class_paras = self.params.get(cst.class_paras, None)
		paras = class_paras.get(self.__class__.__name__) if class_paras is not None else None

		if paras == self.ALL or paras is None:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return inp_list

		if paras == self.NONE:
			self.logger.info("eval_result work. paras is {}".format(paras))
			return None

		if paras:
			self.out_str = paras.split(",")
			self.logger.info("eval_result work. paras is {}".format(self.out_str))
			return [inp_list[int(x)] for x in self.out_str]

	def get_cat_emb(self, cat_features, feat_list, feat_name):
		feat_index = index_of_tensor(self.cat_columns_info.index_of_column, feat_list)
		self.logger.info("cat feature shape {}".format(cat_features.get_shape()))
		cat_feat = tf.gather(cat_features, feat_index, axis=1)
		self.logger.info("cat_feat shape {}".format(cat_feat.get_shape()))

		cat_emb = self.ptable_lookup(list_ids=cat_feat, v_name=feat_name)
		return cat_emb

	def get_dense(self, dense_features, feat_list):
		feat_index = index_of_tensor(self.dense_columns_info.index_of_column, feat_list)
		dense_feat = tf.gather(dense_features, feat_index, axis=1)
		return dense_feat

	def attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, seq_len_fea_col,
	                    embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
	                    att_type):
		with tf.name_scope("attention_layer_%s" % att_type):
			cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
			# 将query复制 seq_len 次 None, seq_len, embed_dim
			if att_type.startswith('top40') or (att_type == 'taobao_din'):
				din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
			elif att_type == 'click_sess_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			elif att_type == 'order_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			else:
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

			activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
			input_layer = din_all

			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
				                             name=name_scope + 'f_%d_att' % i)
				# , reuse=tf.AUTO_REUSE
				input_layer = deep_layer

			din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
			din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30

			# Mask
			key_masks = tf.sequence_mask(seq_len_fea_col, seq_len)  # [B,1, T] 这个已经是三维的了

			paddings = tf.zeros_like(din_output_layer)
			outputs = tf.where(key_masks, din_output_layer, paddings)  # [N, 1, 30]

			tf.summary.histogram("attention", outputs)

			# 直接加权求和
			weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)

			# [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
			weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])  # N, 8*3

			return weighted_outputs

	def attention_layer_nomask(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
	                           embed_dim, seq_fea_num, seq_len, din_deep_layers, din_activation, name_scope,
	                           att_type):
		with tf.name_scope("attention_layer_%s" % att_type):
			self.logger.info("cur_poi_seq_fea_col {}".format(cur_poi_seq_fea_col.get_shape()))
			cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
			# 将query复制 seq_len 次 None, seq_len, embed_dim
			if att_type.startswith('top40'):
				din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
			elif att_type == 'click_sess_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			elif att_type == 'order_att':
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
			else:
				din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

			activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
			input_layer = din_all

			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
				                             name=name_scope + 'f_%d_att' % i)
				# , reuse=tf.AUTO_REUSE
				input_layer = deep_layer

			din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')
			outputs = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, 30

			# Mask
			# key_masks = tf.sequence_mask(seq_len_fea_col, seq_len)  # [B,1, T] 这个已经是三维的了
			#
			# paddings = tf.zeros_like(din_output_layer)
			# outputs = tf.where(key_masks, din_output_layer, paddings)  # [N, 1, 30]
			#
			# tf.summary.histogram("attention", outputs)

			# 直接加权求和
			weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, 30, (N, 30 , 24)= (N, 1, 24)

			# [B,1,seq_len_used]*[B,seq_len_used,seq_fea_num*dim] = [B, 1, seq_fea_num*dim]
			weighted_outputs = tf.reshape(weighted_outputs, [-1, embed_dim * seq_fea_num])  # N, 8*3

			return weighted_outputs

	def tf_print(self, tensor, name):
		shape = tensor.shape
		element = 100
		if len(shape) >= 200:
			element = shape[1].value
		tensor = tf.Print(tensor,
		                  [tensor],
		                  first_n=300,
		                  summarize=element,
		                  message="print_" + name)
		return tensor
