#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor


class meituanMerge(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(meituanMerge, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
        self.target_poi = ["poi_id_int64"]
        self.target_peroid = ["dp_period"]
        self.target_workoroff = ["dp_workoroff"]
        self.seq_len = 300
        self.ucf_complex_poi_list = ["ucf_complex_poi_list[%d]" % i for i in range(self.seq_len)]
        self.ucf_complex_poi_ts_list = ["ucf_complex_poi_ts_list_v2[%d]" % i for i in range(self.seq_len)]
        self.ucf_complex_poi_act_list = ["ucf_complex_poi_act_list[%d]" % i for i in range(self.seq_len)]
        self.ucf_complex_poi_scenario_list = ["ucf_complex_poi_scenario_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_complex_poi_isad_list = ["ucf_complex_poi_isad_list[%d]" % i for i in range(self.seq_len)]
        # self.ucf_poi_tag3_id_list = ["ucf_poi_tag3_id_list[%d]" % i for i in range(self.seq_len)]

        self.cat_list = [self.target_poi, self.target_peroid, self.target_workoroff, self.ucf_complex_poi_list, self.ucf_complex_poi_ts_list, self.ucf_complex_poi_act_list, self.ucf_complex_poi_scenario_list]

        self.ptable_lookup2 = params['ptable_lookup2']
        self.can_input_dim = params['can_input_dim']
        self.can_weight_dim = params['can_weight_dim']
        self.can_bias_dim = params['can_bias_dim']
        self.can_mlp_layer = params['can_mlp_layer']
        self.can_mlp_emb_dim = params['can_mlp_emb_dim']
        self.pw_total_num = self.seq_len

    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas
        self.target_poi, self.target_peroid, self.target_workoroff, self.ucf_complex_poi_list, self.ucf_complex_poi_ts_list, self.ucf_complex_poi_act_list, self.ucf_complex_poi_scenario_list = self.cat_fea_split 
        # self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

        self.ucf_complex_poi_act_list_slot = self.ucf_complex_poi_act_list + (10086 << 44)
        self.ucf_complex_poi_scenario_list_slot = self.ucf_complex_poi_scenario_list + (10087 << 44)
        # self.ucf_poi_tag3_id_list_slot = self.ucf_poi_tag3_id_list + (10088 << 44)
        # self.ucf_complex_poi_isad_list_slot = self.ucf_complex_poi_isad_list + (10089 << 44)

        self.target_poi_emb, self.ucf_complex_poi_list_emb = \
            self.ptable_lookup(list_ids=[self.target_poi, self.ucf_complex_poi_list], v_name=self.__class__.__name__)
        # self.ucf_complex_poi_act_list_slot, self.ucf_complex_poi_scenario_list_slot, self.ucf_complex_poi_isad_list_slot, self.ucf_poi_tag3_id_list_slot
       
    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        trace_timestamp_hour = (self.ucf_complex_poi_ts_list - 1609689600) / 3600 # 1609689600 是 2021-01-04 00:00:00, 是周一
        trace_week = trace_timestamp_hour / 24 % 7
        trace_hour = trace_timestamp_hour % 24
        trace_time_interval = trace_timestamp_hour - tf.expand_dims(trace_timestamp_hour[:, 0], 1)

        trace_time_feature = tf.stack([trace_week, trace_hour, trace_time_interval], axis=-1) # [B, 300, 3]
        trace_time_feature = tf.cast(trace_time_feature, tf.float32)
        imputation_w = tf.get_variable('imputation_w', shape=[1, 1, 3, 8], initializer=tf.random_normal_initializer())
        trace_time_feature_emb = tf.expand_dims(trace_time_feature, axis=3) * imputation_w # [B, 300, 3, 8]
        trace_time_feature_emb = tf.reshape(trace_time_feature_emb, [-1, self.pw_total_num, 3*8])

        # export_input = tf.concat([trace_time_feature_emb, self.ucf_complex_poi_list_emb, self.ucf_poi_tag3_id_list_emb, self.ucf_complex_poi_scenario_list_emb, self.ucf_complex_poi_act_list_emb], -1)
        export_input = self.ucf_complex_poi_list_emb
        mask = tf.where(tf.equal(self.ucf_complex_poi_list, 0), tf.zeros_like(self.ucf_complex_poi_list), tf.ones_like(self.ucf_complex_poi_list)) # [B, seq]

        trace_weekoroff = tf.where(tf.less(trace_week, 5), tf.ones_like(trace_week), tf.zeros_like(trace_week))
        trace_weekoroff = (trace_weekoroff + 1) * mask
        trace_period = tf.zeros_like(trace_hour)
        trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 10), tf.less(trace_hour, 14)), 1 * tf.ones_like(trace_period), trace_period)
        trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 14), tf.less(trace_hour, 17)), 2 * tf.ones_like(trace_period), trace_period)
        trace_period = tf.where(tf.logical_and(tf.greater_equal(trace_hour, 17), tf.less(trace_hour, 21)), 3 * tf.ones_like(trace_period), trace_period)
        trace_period = tf.where(tf.logical_or(tf.greater_equal(trace_hour, 21), tf.less(trace_hour, 5)), 4 * tf.ones_like(trace_period), trace_period)
        trace_period = (trace_period + 1) * mask

        ## input
        tgt_scene_list = [self.target_peroid + 1, self.target_workoroff + 1, 2 * tf.ones_like(self.target_workoroff)] 
        seq_scene_list = [trace_period, trace_weekoroff, self.ucf_complex_poi_act_list]

        slot_list = [11111, 22222, 33333]
        tgt_scene_list = [scene + (slot << 44) for scene, slot in zip(tgt_scene_list, slot_list)]
        seq_scene_list = [scene + (slot << 44) for scene, slot in zip(seq_scene_list, slot_list)]

        scene_set_list = [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3]]
        scene_set_list = [[x_ + (y << 44) for x_ in x] for x, y in zip(scene_set_list, slot_list)]

        tgt_emb = self.target_poi_emb
        seq_emb = export_input

        def get_scene_params_list(scene_list):
            with tf.variable_scope('get_scene_params_list', reuse=tf.AUTO_REUSE):
                share_export_num = 1
                specific_export_num = 1
                slot = 0
                scene_params_list = [self.get_scene_params(scene, slot, share_export_num, specific_export_num, name=str(i)) for i, scene in enumerate(scene_list)] # [B, seq, param_dim] * 3
                scene_param = tf.stack(scene_params_list, -1) # [B, seq, param_dim, param_num]
                param_weight = tf.get_variable('param_weight', shape=[len(scene_params_list), 1], initializer=tf.random_normal_initializer())
                scene_param_merge = tf.squeeze(tf.matmul(scene_param, param_weight), -1) # [B, seq, param_dim]
                scene_params_list.append(scene_param_merge)
                return scene_params_list

        tgt_scene_params_list = get_scene_params_list(tgt_scene_list)
        seq_scene_params_list = get_scene_params_list(seq_scene_list)

        specific_scene_emb = [self.specific_scene_din2(tgt_emb, seq_emb, tgt_scene_params_list[i], seq_scene_params_list[i], mask, name=str(i)) for i in range(len(tgt_scene_list))]
        merge_scene_emb = self.specific_scene_din2(tgt_emb, seq_emb, tgt_scene_params_list[-1], seq_scene_params_list[-1], mask, name='merge') 

        def scene_emb_merge(specific_scene_emb_list, merge_scene_emb):
            merge_weight = tf.get_variable('merge_weight', shape=[1, len(specific_scene_emb_list), 1], initializer=tf.random_normal_initializer())
            specific_scene_emb = tf.stack(specific_scene_emb_list, 1) # [B, scene_num, emb]

            scene_similarity = tf.matmul(specific_scene_emb, merge_scene_emb[:, :, None]) # [B, scene_num, 1]
            scene_score = merge_weight + scene_similarity

            output_merge = tf.reduce_mean(specific_scene_emb * scene_score, 1) # [B, emb]
            return output_merge


        merge_scene_emb2 = scene_emb_merge(specific_scene_emb, merge_scene_emb)
        output_p = tf.concat([merge_scene_emb, merge_scene_emb2], -1)
        return output_p
        
        
       

    def specific_scene_din2(self, target_poi_emb, seq_poi_emb, target_scene_param, trace_scene_param, seq_mask, name):
        with tf.variable_scope('specific_scene_din_' + name, reuse=tf.AUTO_REUSE):
            # seq_mask = tf.where(tf.equal(target_scene, trace_scene), tf.ones_like(trace_scene), tf.zeros_like(trace_scene))
            # seq_mask = tf.Print(seq_mask, ["WSL:seq_mask", seq_mask], first_n=100, summarize=100)

            target_poi_emb = self.can_unit(target_poi_emb, target_scene_param)
            seq_poi_emb = self.can_unit(seq_poi_emb, trace_scene_param)

            target_poi_emb_tile = tf.tile(target_poi_emb, [1, self.seq_len, 1])
            din_input = tf.concat([target_poi_emb_tile, seq_poi_emb], -1) # [B, 300, 16]
            att = self.my_dense(din_input, 32, 'specific_scene_din_' + name + 'din_first_layer')
            att = self.my_dense(att, 1, 'specific_scene_din_' + name + 'din_second_layer') # [B, 300, 1]
            att = att * tf.cast(seq_mask[:, :, None], tf.float32)

            output = tf.reduce_mean(att * seq_poi_emb, 1) #[B, 8]
            self.logger.info("#WSL: specific_scene_din_{} is {}".format(name, output))
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
            