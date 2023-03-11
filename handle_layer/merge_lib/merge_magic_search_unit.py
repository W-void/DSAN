#!/usr/bin/python
# -*- coding:utf-8 -*-

# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor


class MagicSearch(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(MagicSearch, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["murmurhash_poi_id_int64"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]

		def create_dense(name, input_layer):
			din_deep_layers = [256, 64, 8]
			name_scope = "create_dense"
			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.relu,
				                             name="_".join([name_scope, name, 'dense_%d' % i]))
				input_layer = deep_layer
			return input_layer

		merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]
		self.logger.info("merge_cand {}".format(merge_cand))
		merge_cand = tf.reshape(merge_cand, [-1, feat_len, embed_dim])
		self.logger.info("merge_cand {}".format(merge_cand.get_shape()))
		# merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]

		output = self.attention_layer_nomask(self.target_poi_emb, merge_cand, embed_dim, 1, feat_len, self.params['din_deep_layers'],
		                                     self.params['din_activation'], key, key + '_att')

		return output  # [B, 3*8]


class MagicOutDin(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(MagicOutDin, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.target_poi = ["murmurhash_poi_id_int64"]
		self.cat_list = [self.target_poi]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)
		self.target_poi_emb = self.cat_fea_split_emb[-1]

	def __call__(self, model_classes, process_features, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		target_cate = self.target_poi
		feat_len = len(model_classes)
		key = self.__class__.__name__
		input_names = [m.__class__.__name__ for m in model_classes]
		din_deep_layers = [128, 64]
		self.target_poi_emb = tf.reshape(self.target_poi_emb, [-1, 1, self.base_embed_dim])

		def create_dense(name, input_layer):

			name_scope = "create_dense"
			for i in range(len(din_deep_layers)):
				deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=tf.nn.relu,
				                             name="_".join([name_scope, name, 'dense_%d' % i]))
				input_layer = deep_layer
			return input_layer

		merge_cand = [create_dense(name, inp) for name, inp in zip(input_names, process_features)]
		rs = []
		for i, cand in enumerate(merge_cand):
			his_cand = merge_cand[:i] + merge_cand[i + 1:]
			cur_cand = cand

			self.logger.info("his_cand len {}".format(len(his_cand)))

			cur_cand = tf.reshape(cur_cand, [-1, 1, din_deep_layers[-1]])
			his_cand = tf.reshape(his_cand, [-1, feat_len - 1, din_deep_layers[-1]])

			self.logger.info("cur_cand {}".format(cur_cand.get_shape()))
			self.logger.info("his_cand {}".format(his_cand.get_shape()))
			output = self.attention_layer_nomask(cur_cand, his_cand, din_deep_layers[-1], 1, feat_len - 1, self.params['din_deep_layers'],
			                                     self.params['din_activation'], "_".join([key, str(i)]), key + '_att')
			rs.append(output)

		return tf.concat(rs, axis=1)  # [B, 3*8]
