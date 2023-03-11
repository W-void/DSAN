#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor


class taobaoMerge(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(taobaoMerge, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
        self.cat_fea_list = ["adgroup_id", "pid", "cate_id", "customer_id", "brand", "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level", "shopping_level", "occupation", "new_user_class_level"]
        self.tgt_time = ["time_stamp"]
        self.clk_num = 50
        self.clkFeaTimeStampList = ["clk_seq_tim_stamp[%d]" % i for i in range(self.clk_num)]
        self.clkFeaBehaviorTagList = ["clk_seq_btag[%d]" % i for i in range(self.clk_num)]
        self.clkFeaCateList = ["clk_seq_cate[%d]" % i for i in range(self.clk_num)]
        self.clkFeaBandList = ["clk_seq_brand[%d]" % i for i in range(self.clk_num)]
        self.seq_num = 300
        self.feaTimeStampList = ["seq_tim_stamp[%d]" % i for i in range(self.seq_num)]
        self.feaBehaviorTagList = ["seq_btag[%d]" % i for i in range(self.seq_num)]
        self.feaCateList = ["seq_cate[%d]" % i for i in range(self.seq_num)]
        self.feaBandList = ["seq_brand[%d]" % i for i in range(self.seq_num)]

        self.clk_seq_list = [self.clkFeaTimeStampList, self.clkFeaBehaviorTagList, self.clkFeaCateList, self.clkFeaBandList]
        self.all_seq_list = [self.feaTimeStampList, self.feaBehaviorTagList, self.feaCateList, self.feaBandList]
        self.cat_list = [self.tgt_time, self.cat_fea_list] + self.all_seq_list
        self.dense_list = [["price"]]
        # self.target_poi = ["poi_id_int64"]
        # self.target_peroid = ["dp_period"]
        # self.target_workoroff = ["dp_workoroff"]
        # self.seq_len = 300
        # self.ucf_complex_poi_list = ["ucf_complex_poi_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_complex_poi_ts_list = ["ucf_complex_poi_ts_list_v2[%d]" % i for i in range(self.seq_len)]
        # self.ucf_complex_poi_act_list = ["ucf_complex_poi_act_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_complex_poi_scenario_list = ["ucf_complex_poi_scenario_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_complex_poi_isad_list = ["ucf_complex_poi_isad_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_poi_tag3_id_list = ["ucf_poi_tag3_id_list[%d]" % i for i in range(self.seq_len)]

        # self.cat_list = [self.target_poi, self.target_peroid, self.target_workoroff, self.ucf_complex_poi_list, self.ucf_complex_poi_ts_list, self.ucf_complex_poi_act_list, self.ucf_complex_poi_scenario_list, self.ucf_complex_poi_isad_list, self.ucf_poi_tag3_id_list]

        self.ptable_lookup2 = params['ptable_lookup2']
        self.can_input_dim = params['can_input_dim']
        self.can_weight_dim = params['can_weight_dim']
        self.can_bias_dim = params['can_bias_dim']
        self.can_mlp_layer = params['can_mlp_layer']
        self.can_mlp_emb_dim = params['can_mlp_emb_dim']
        self.pw_total_num = self.seq_num

    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas
        self.tgt_time, self.cat_fea_list, self.clkFeaTimeStampList, self.clkFeaBehaviorTagList, self.clkFeaCateList, self.clkFeaBandList = self.cat_fea_split
        self.cat_fea_emb = self.ptable_lookup(list_ids=[self.cat_fea_list], v_name=self.__class__.__name__)[0]
        self.clk_brand_emb = self.ptable_lookup(list_ids=[self.clkFeaBandList], v_name=self.__class__.__name__)[0]
        self.clk_cate_emb = self.ptable_lookup(list_ids=[self.clkFeaCateList], v_name=self.__class__.__name__)[0]
        self.clk_btag_emb = self.ptable_lookup(list_ids=[self.clkFeaBehaviorTagList], v_name=self.__class__.__name__)[0]
        self.logger.info("self.cat_fea_emb : {}".format(self.cat_fea_emb))
        self.logger.info("self.clk_brand_emb : {}".format(self.clk_brand_emb))
        self.target_brand_emb = self.cat_fea_emb[:, 4]
        self.adgroup_id_emb = self.cat_fea_emb[:, 0]
        self.cate_id_emb = self.cat_fea_emb[:, 2]
        
    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        tgt_comb = tf.concat([self.target_brand_emb[:, None], self.cate_id_emb[:, None], tf.zeros_like(self.cate_id_emb[:, None])], -1)
        seq_comb = tf.concat([self.clk_brand_emb, self.clk_cate_emb, self.clk_btag_emb], -1)
        mask = tf.count_nonzero(self.clkFeaTimeStampList, -1, keep_dims=True)
        din_output = self.attention_layer(tgt_comb, seq_comb, mask,
                                            embed_dim, 3, self.seq_num,
                                            self.params['din_deep_layers'], self.params['din_activation'],
                                            'taobao_din', 'taobao_din')
        
        
        def get_time_scene(input):
            trace_timestamp_hour = (input - 1609689600) / 3600 # 1609689600 是 2021-01-04 00:00:00, 是周一
            trace_week = trace_timestamp_hour / 24 % 7
            trace_hour = trace_timestamp_hour % 24

            trace_weekoroff = tf.where(tf.less(trace_week, 5), tf.ones_like(trace_week), tf.zeros_like(trace_week))
            trace_period = tf.zeros_like(trace_hour)
            trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 10), tf.less(trace_hour, 14)), 1 * tf.ones_like(trace_period), trace_period)
            trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 14), tf.less(trace_hour, 17)), 2 * tf.ones_like(trace_period), trace_period)
            trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 17), tf.less(trace_hour, 21)), 3 * tf.ones_like(trace_period), trace_period)
            trace_period = tf.where(tf.logical_or(tf.greater_equal(trace_hour, 21), tf.less(trace_hour, 5)), 4 * tf.ones_like(trace_period), trace_period)

            return [trace_weekoroff, trace_period]

        tgt_scene_list = get_time_scene(self.tgt_time) + [tf.ones_like(self.tgt_time) * (1001<<44)]
        seq_scene_list = get_time_scene(self.clkFeaTimeStampList) + [self.clkFeaBehaviorTagList]

        ## DIF
        tgt_scene_emb_list = [self.ptable_lookup(list_ids=[ids], v_name=self.__class__.__name__)[0] for ids in tgt_scene_list] # [B, 1, 8]
        seq_scene_emb_list = [self.ptable_lookup(list_ids=[ids], v_name=self.__class__.__name__)[0] for ids in seq_scene_list] # [B, 300, 8]

        tgt_scene_emb = tf.stack(tgt_scene_emb_list, 1) # [B, 2, 1, 8]
        seq_scene_emb = tf.stack(seq_scene_emb_list, 1) # [B, 2, 300, 8]

        att_scene = tf.matmul(tgt_scene_emb, tf.transpose(seq_scene_emb, [0, 1, 3, 2])) # [B, 2, 1, 300]
        param_weight = tf.get_variable('param_weight_dif', shape=[len(tgt_scene_emb_list), 1], initializer=tf.random_normal_initializer())
        att_scene_merge = tf.matmul(tf.transpose(att_scene, [0, 2, 3, 1]), param_weight) # [B, 1, 300, 1]
        output_dif = tf.reduce_mean(tf.squeeze(att_scene_merge, 1) * self.target_brand_emb[:, None], 1) # [B, 8]
        ## DIF end

        tgt_scene_param_list = [self.get_scene_params(scene, 0, 1, 1, 'tgt_scene_param') for scene in tgt_scene_list]
        seq_scene_param_list = [self.get_scene_params(scene, 0, 1, 1, 'seq_scene_param') for scene in seq_scene_list]

        tgt_scene_param = tf.stack(tgt_scene_param_list, -1) # [B, 1, param_dim, param_num]
        seq_scene_param = tf.stack(seq_scene_param_list, -1) # [B, seq, param_dim, param_num]

        param_weight = tf.get_variable('param_weight', shape=[len(tgt_scene_param_list), 1], initializer=tf.random_normal_initializer())
        seq_scene_param_merge = tf.squeeze(tf.matmul(seq_scene_param, param_weight), -1) # [B, seq, param_dim]
        tgt_scene_param_merge = tf.squeeze(tf.matmul(tgt_scene_param, param_weight), -1) # [B, 1, param_dim]

        tgt_scene_param_list.append(tgt_scene_param_merge)
        seq_scene_param_list.append(seq_scene_param_merge)

        output_dsan = []
        for i in range(len(tgt_scene_param_list)):
            tgt_scene_param, seq_scene_param = tgt_scene_param_list[i], seq_scene_param_list[i]
            seq_scene_output = self.can_unit(self.clk_brand_emb, seq_scene_param)
            tgt_scene_output = self.can_unit(self.target_brand_emb[:, None], tgt_scene_param)

            output_scene_din = self.attention_layer(tgt_scene_output, seq_scene_output, mask,
                                                    embed_dim, 1, self.seq_num,
                                                    self.params['din_deep_layers'], self.params['din_activation'],
                                                    'output_scene_din%d'%i, 'output_scene_din%d'%i)
            output_dsan.append(output_scene_din)

        def scene_emb_merge(scene_emb_list):
            specific_scene_emb_list, merge_scene_emb = scene_emb_list[:-1], scene_emb_list[-1]
            merge_weight = tf.get_variable('merge_weight', shape=[1, len(specific_scene_emb_list), 1], initializer=tf.random_normal_initializer())
            specific_scene_emb = tf.stack(specific_scene_emb_list, 1) # [B, scene_num, emb]

            scene_similarity = tf.matmul(specific_scene_emb, merge_scene_emb[:, :, None]) # [B, scene_num, 1]
            scene_score = merge_weight + scene_similarity

            output_merge = tf.reduce_mean(specific_scene_emb * scene_score, 1) # [B, emb]
            return output_merge

        output_dsan_merge = scene_emb_merge(output_dsan)

        output = tf.concat(output_dsan + [output_dsan_merge, tf.layers.flatten(tgt_scene_output), output_dif, tf.layers.flatten(self.cat_fea_emb), din_output], -1)
        return output

    def my_dense(self, inputs, output_dims, name, activation=tf.nn.relu):
        with tf.variable_scope('dense_layer_' + name, reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=inputs, units=output_dims,
                                    kernel_initializer=tf.glorot_normal_initializer(),
                                    bias_initializer=tf.glorot_normal_initializer(),
                                    activation=activation, name='dense_layer_' + name)
            return output
    
    def can_unit(self, input_feature, can_params):
        can_input = input_feature[:, :, None, :]
        batch_size = input_feature.shape[0]
        trace_scene_list_emb_reshape = tf.reshape(can_params, [batch_size, -1, self.can_mlp_layer, self.can_weight_dim + self.can_bias_dim])    
        for i in range(self.can_mlp_layer):
            cur_can_weight = tf.reshape(trace_scene_list_emb_reshape[:, :, i, :self.can_weight_dim], [batch_size, -1, self.can_input_dim, self.can_input_dim])
            cur_can_bias = tf.reshape(trace_scene_list_emb_reshape[:, :, i, self.can_weight_dim:], [batch_size, -1, 1, self.can_input_dim])
            cur_output = tf.matmul(can_input, cur_can_weight) # [B, 300, 1, 8]
            cur_output = tf.math.add(cur_output, cur_can_bias) # [B, 300, 1, 8]
            cur_output = tf.nn.tanh(cur_output)
            can_input = cur_output # [B, 300, 1, 8]
            self.logger.info('#WSL: cur_output %s', cur_output)
        can_output = tf.reshape(cur_output, [batch_size, -1, self.can_input_dim]) # [B, 300, 8]
        return can_output
    
    def get_scene_params(self, scene_id_list, slot, share_export_num, specific_export_num, name):
        with tf.variable_scope('get_scene_params' + name, reuse=tf.AUTO_REUSE):
            # share_tower = [self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + i, v_name=self.__class__.__name__+'mmoe_unit_lookup_'+name) for i in range(share_export_num)]
            # specific_tower = [self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + share_export_num + 10 * i, v_name=self.__class__.__name__+'mmoe_unit_lookup_'+name) for i in range(specific_export_num)]
            share_tower = self.ptable_lookup2(list_ids=tf.zeros_like(scene_id_list) + (slot << 44), v_name=self.__class__.__name__+'lookup_share_tower')[0]
            specific_tower = self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + 1, v_name=self.__class__.__name__+'lookup_specific_tower')[0]
            return share_tower + specific_tower
            