#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor


class baseDin(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(baseDin, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
        self.cat_fea_list = ["user_id", "gender", "visit_city", "is_supervip", "ctr_30", "ord_30", "shop_id", "item_id", "city_id", "district_id", "shop_aoi_id", "shop_geohash_6", "shop_geohash_12", "brand_id", "category_1_id", "merge_standard_food_id", "rank_7", "rank_30", "rank_90", "times", "hours", "time_type", "weekdays", "geohash12"]
        self.cat_seq_fea = ["shop_id_list[0]", "shop_id_list[1]", "shop_id_list[2]", "shop_id_list[3]", "shop_id_list[4]", "shop_id_list[5]", "shop_id_list[6]", "shop_id_list[7]", "shop_id_list[8]", "shop_id_list[9]", "shop_id_list[10]", "shop_id_list[11]", "shop_id_list[12]", "shop_id_list[13]", "shop_id_list[14]", "shop_id_list[15]", "shop_id_list[16]", "shop_id_list[17]", "shop_id_list[18]", "shop_id_list[19]", "shop_id_list[20]", "shop_id_list[21]", "shop_id_list[22]", "shop_id_list[23]", "shop_id_list[24]", "shop_id_list[25]", "shop_id_list[26]", "shop_id_list[27]", "shop_id_list[28]", "shop_id_list[29]", "shop_id_list[30]", "shop_id_list[31]", "shop_id_list[32]", "shop_id_list[33]", "shop_id_list[34]", "shop_id_list[35]", "shop_id_list[36]", "shop_id_list[37]", "shop_id_list[38]", "shop_id_list[39]", "shop_id_list[40]", "shop_id_list[41]", "shop_id_list[42]", "shop_id_list[43]", "shop_id_list[44]", "shop_id_list[45]", "shop_id_list[46]", "shop_id_list[47]", "shop_id_list[48]", "shop_id_list[49]",
                            "item_id_list[0]", "item_id_list[1]", "item_id_list[2]", "item_id_list[3]", "item_id_list[4]", "item_id_list[5]", "item_id_list[6]", "item_id_list[7]", "item_id_list[8]", "item_id_list[9]", "item_id_list[10]", "item_id_list[11]", "item_id_list[12]", "item_id_list[13]", "item_id_list[14]", "item_id_list[15]", "item_id_list[16]", "item_id_list[17]", "item_id_list[18]", "item_id_list[19]", "item_id_list[20]", "item_id_list[21]", "item_id_list[22]", "item_id_list[23]", "item_id_list[24]", "item_id_list[25]", "item_id_list[26]", "item_id_list[27]", "item_id_list[28]", "item_id_list[29]", "item_id_list[30]", "item_id_list[31]", "item_id_list[32]", "item_id_list[33]", "item_id_list[34]", "item_id_list[35]", "item_id_list[36]", "item_id_list[37]", "item_id_list[38]", "item_id_list[39]", "item_id_list[40]", "item_id_list[41]", "item_id_list[42]", "item_id_list[43]", "item_id_list[44]", "item_id_list[45]", "item_id_list[46]", "item_id_list[47]", "item_id_list[48]", "item_id_list[49]",
                            "category_1_id_list[0]", "category_1_id_list[1]", "category_1_id_list[2]", "category_1_id_list[3]", "category_1_id_list[4]", "category_1_id_list[5]", "category_1_id_list[6]", "category_1_id_list[7]", "category_1_id_list[8]", "category_1_id_list[9]", "category_1_id_list[10]", "category_1_id_list[11]", "category_1_id_list[12]", "category_1_id_list[13]", "category_1_id_list[14]", "category_1_id_list[15]", "category_1_id_list[16]", "category_1_id_list[17]", "category_1_id_list[18]", "category_1_id_list[19]", "category_1_id_list[20]", "category_1_id_list[21]", "category_1_id_list[22]", "category_1_id_list[23]", "category_1_id_list[24]", "category_1_id_list[25]", "category_1_id_list[26]", "category_1_id_list[27]", "category_1_id_list[28]", "category_1_id_list[29]", "category_1_id_list[30]", "category_1_id_list[31]", "category_1_id_list[32]", "category_1_id_list[33]", "category_1_id_list[34]", "category_1_id_list[35]", "category_1_id_list[36]", "category_1_id_list[37]", "category_1_id_list[38]", "category_1_id_list[39]", "category_1_id_list[40]", "category_1_id_list[41]", "category_1_id_list[42]", "category_1_id_list[43]", "category_1_id_list[44]", "category_1_id_list[45]", "category_1_id_list[46]", "category_1_id_list[47]", "category_1_id_list[48]", "category_1_id_list[49]", 
                            "merge_standard_food_id_list[0]", "merge_standard_food_id_list[1]", "merge_standard_food_id_list[2]", "merge_standard_food_id_list[3]", "merge_standard_food_id_list[4]", "merge_standard_food_id_list[5]", "merge_standard_food_id_list[6]", "merge_standard_food_id_list[7]", "merge_standard_food_id_list[8]", "merge_standard_food_id_list[9]", "merge_standard_food_id_list[10]", "merge_standard_food_id_list[11]", "merge_standard_food_id_list[12]", "merge_standard_food_id_list[13]", "merge_standard_food_id_list[14]", "merge_standard_food_id_list[15]", "merge_standard_food_id_list[16]", "merge_standard_food_id_list[17]", "merge_standard_food_id_list[18]", "merge_standard_food_id_list[19]", "merge_standard_food_id_list[20]", "merge_standard_food_id_list[21]", "merge_standard_food_id_list[22]", "merge_standard_food_id_list[23]", "merge_standard_food_id_list[24]", "merge_standard_food_id_list[25]", "merge_standard_food_id_list[26]", "merge_standard_food_id_list[27]", "merge_standard_food_id_list[28]", "merge_standard_food_id_list[29]", "merge_standard_food_id_list[30]", "merge_standard_food_id_list[31]", "merge_standard_food_id_list[32]", "merge_standard_food_id_list[33]", "merge_standard_food_id_list[34]", "merge_standard_food_id_list[35]", "merge_standard_food_id_list[36]", "merge_standard_food_id_list[37]", "merge_standard_food_id_list[38]", "merge_standard_food_id_list[39]", "merge_standard_food_id_list[40]", "merge_standard_food_id_list[41]", "merge_standard_food_id_list[42]", "merge_standard_food_id_list[43]", "merge_standard_food_id_list[44]", "merge_standard_food_id_list[45]", "merge_standard_food_id_list[46]", "merge_standard_food_id_list[47]", "merge_standard_food_id_list[48]", "merge_standard_food_id_list[49]", 
                            "brand_id_list[0]", "brand_id_list[1]", "brand_id_list[2]", "brand_id_list[3]", "brand_id_list[4]", "brand_id_list[5]", "brand_id_list[6]", "brand_id_list[7]", "brand_id_list[8]", "brand_id_list[9]", "brand_id_list[10]", "brand_id_list[11]", "brand_id_list[12]", "brand_id_list[13]", "brand_id_list[14]", "brand_id_list[15]", "brand_id_list[16]", "brand_id_list[17]", "brand_id_list[18]", "brand_id_list[19]", "brand_id_list[20]", "brand_id_list[21]", "brand_id_list[22]", "brand_id_list[23]", "brand_id_list[24]", "brand_id_list[25]", "brand_id_list[26]", "brand_id_list[27]", "brand_id_list[28]", "brand_id_list[29]", "brand_id_list[30]", "brand_id_list[31]", "brand_id_list[32]", "brand_id_list[33]", "brand_id_list[34]", "brand_id_list[35]", "brand_id_list[36]", "brand_id_list[37]", "brand_id_list[38]", "brand_id_list[39]", "brand_id_list[40]", "brand_id_list[41]", "brand_id_list[42]", "brand_id_list[43]", "brand_id_list[44]", "brand_id_list[45]", "brand_id_list[46]", "brand_id_list[47]", "brand_id_list[48]", "brand_id_list[49]", 
                            "shop_aoi_id_list[0]", "shop_aoi_id_list[1]", "shop_aoi_id_list[2]", "shop_aoi_id_list[3]", "shop_aoi_id_list[4]", "shop_aoi_id_list[5]", "shop_aoi_id_list[6]", "shop_aoi_id_list[7]", "shop_aoi_id_list[8]", "shop_aoi_id_list[9]", "shop_aoi_id_list[10]", "shop_aoi_id_list[11]", "shop_aoi_id_list[12]", "shop_aoi_id_list[13]", "shop_aoi_id_list[14]", "shop_aoi_id_list[15]", "shop_aoi_id_list[16]", "shop_aoi_id_list[17]", "shop_aoi_id_list[18]", "shop_aoi_id_list[19]", "shop_aoi_id_list[20]", "shop_aoi_id_list[21]", "shop_aoi_id_list[22]", "shop_aoi_id_list[23]", "shop_aoi_id_list[24]", "shop_aoi_id_list[25]", "shop_aoi_id_list[26]", "shop_aoi_id_list[27]", "shop_aoi_id_list[28]", "shop_aoi_id_list[29]", "shop_aoi_id_list[30]", "shop_aoi_id_list[31]", "shop_aoi_id_list[32]", "shop_aoi_id_list[33]", "shop_aoi_id_list[34]", "shop_aoi_id_list[35]", "shop_aoi_id_list[36]", "shop_aoi_id_list[37]", "shop_aoi_id_list[38]", "shop_aoi_id_list[39]", "shop_aoi_id_list[40]", "shop_aoi_id_list[41]", "shop_aoi_id_list[42]", "shop_aoi_id_list[43]", "shop_aoi_id_list[44]", "shop_aoi_id_list[45]", "shop_aoi_id_list[46]", "shop_aoi_id_list[47]", "shop_aoi_id_list[48]", "shop_aoi_id_list[49]", 
                            "shop_geohash6_list[0]", "shop_geohash6_list[1]", "shop_geohash6_list[2]", "shop_geohash6_list[3]", "shop_geohash6_list[4]", "shop_geohash6_list[5]", "shop_geohash6_list[6]", "shop_geohash6_list[7]", "shop_geohash6_list[8]", "shop_geohash6_list[9]", "shop_geohash6_list[10]", "shop_geohash6_list[11]", "shop_geohash6_list[12]", "shop_geohash6_list[13]", "shop_geohash6_list[14]", "shop_geohash6_list[15]", "shop_geohash6_list[16]", "shop_geohash6_list[17]", "shop_geohash6_list[18]", "shop_geohash6_list[19]", "shop_geohash6_list[20]", "shop_geohash6_list[21]", "shop_geohash6_list[22]", "shop_geohash6_list[23]", "shop_geohash6_list[24]", "shop_geohash6_list[25]", "shop_geohash6_list[26]", "shop_geohash6_list[27]", "shop_geohash6_list[28]", "shop_geohash6_list[29]", "shop_geohash6_list[30]", "shop_geohash6_list[31]", "shop_geohash6_list[32]", "shop_geohash6_list[33]", "shop_geohash6_list[34]", "shop_geohash6_list[35]", "shop_geohash6_list[36]", "shop_geohash6_list[37]", "shop_geohash6_list[38]", "shop_geohash6_list[39]", "shop_geohash6_list[40]", "shop_geohash6_list[41]", "shop_geohash6_list[42]", "shop_geohash6_list[43]", "shop_geohash6_list[44]", "shop_geohash6_list[45]", "shop_geohash6_list[46]", "shop_geohash6_list[47]", "shop_geohash6_list[48]", "shop_geohash6_list[49]",
                            "timediff_list[0]", "timediff_list[1]", "timediff_list[2]", "timediff_list[3]", "timediff_list[4]", "timediff_list[5]", "timediff_list[6]", "timediff_list[7]", "timediff_list[8]", "timediff_list[9]", "timediff_list[10]", "timediff_list[11]", "timediff_list[12]", "timediff_list[13]", "timediff_list[14]", "timediff_list[15]", "timediff_list[16]", "timediff_list[17]", "timediff_list[18]", "timediff_list[19]", "timediff_list[20]", "timediff_list[21]", "timediff_list[22]", "timediff_list[23]", "timediff_list[24]", "timediff_list[25]", "timediff_list[26]", "timediff_list[27]", "timediff_list[28]", "timediff_list[29]", "timediff_list[30]", "timediff_list[31]", "timediff_list[32]", "timediff_list[33]", "timediff_list[34]", "timediff_list[35]", "timediff_list[36]", "timediff_list[37]", "timediff_list[38]", "timediff_list[39]", "timediff_list[40]", "timediff_list[41]", "timediff_list[42]", "timediff_list[43]", "timediff_list[44]", "timediff_list[45]", "timediff_list[46]", "timediff_list[47]", "timediff_list[48]", "timediff_list[49]", 
                            "hours_list[0]", "hours_list[1]", "hours_list[2]", "hours_list[3]", "hours_list[4]", "hours_list[5]", "hours_list[6]", "hours_list[7]", "hours_list[8]", "hours_list[9]", "hours_list[10]", "hours_list[11]", "hours_list[12]", "hours_list[13]", "hours_list[14]", "hours_list[15]", "hours_list[16]", "hours_list[17]", "hours_list[18]", "hours_list[19]", "hours_list[20]", "hours_list[21]", "hours_list[22]", "hours_list[23]", "hours_list[24]", "hours_list[25]", "hours_list[26]", "hours_list[27]", "hours_list[28]", "hours_list[29]", "hours_list[30]", "hours_list[31]", "hours_list[32]", "hours_list[33]", "hours_list[34]", "hours_list[35]", "hours_list[36]", "hours_list[37]", "hours_list[38]", "hours_list[39]", "hours_list[40]", "hours_list[41]", "hours_list[42]", "hours_list[43]", "hours_list[44]", "hours_list[45]", "hours_list[46]", "hours_list[47]", "hours_list[48]", "hours_list[49]", 
                            "time_type_list[0]", "time_type_list[1]", "time_type_list[2]", "time_type_list[3]", "time_type_list[4]", "time_type_list[5]", "time_type_list[6]", "time_type_list[7]", "time_type_list[8]", "time_type_list[9]", "time_type_list[10]", "time_type_list[11]", "time_type_list[12]", "time_type_list[13]", "time_type_list[14]", "time_type_list[15]", "time_type_list[16]", "time_type_list[17]", "time_type_list[18]", "time_type_list[19]", "time_type_list[20]", "time_type_list[21]", "time_type_list[22]", "time_type_list[23]", "time_type_list[24]", "time_type_list[25]", "time_type_list[26]", "time_type_list[27]", "time_type_list[28]", "time_type_list[29]", "time_type_list[30]", "time_type_list[31]", "time_type_list[32]", "time_type_list[33]", "time_type_list[34]", "time_type_list[35]", "time_type_list[36]", "time_type_list[37]", "time_type_list[38]", "time_type_list[39]", "time_type_list[40]", "time_type_list[41]", "time_type_list[42]", "time_type_list[43]", "time_type_list[44]", "time_type_list[45]", "time_type_list[46]", "time_type_list[47]", "time_type_list[48]", "time_type_list[49]", 
                            "weekdays_list[0]", "weekdays_list[1]", "weekdays_list[2]", "weekdays_list[3]", "weekdays_list[4]", "weekdays_list[5]", "weekdays_list[6]", "weekdays_list[7]", "weekdays_list[8]", "weekdays_list[9]", "weekdays_list[10]", "weekdays_list[11]", "weekdays_list[12]", "weekdays_list[13]", "weekdays_list[14]", "weekdays_list[15]", "weekdays_list[16]", "weekdays_list[17]", "weekdays_list[18]", "weekdays_list[19]", "weekdays_list[20]", "weekdays_list[21]", "weekdays_list[22]", "weekdays_list[23]", "weekdays_list[24]", "weekdays_list[25]", "weekdays_list[26]", "weekdays_list[27]", "weekdays_list[28]", "weekdays_list[29]", "weekdays_list[30]", "weekdays_list[31]", "weekdays_list[32]", "weekdays_list[33]", "weekdays_list[34]", "weekdays_list[35]", "weekdays_list[36]", "weekdays_list[37]", "weekdays_list[38]", "weekdays_list[39]", "weekdays_list[40]", "weekdays_list[41]", "weekdays_list[42]", "weekdays_list[43]", "weekdays_list[44]", "weekdays_list[45]", "weekdays_list[46]", "weekdays_list[47]", "weekdays_list[48]", "weekdays_list[49]"
                            ]
        self.dense_fea = ["avg_price", "total_amt_30"]
        self.dense_seq_fea = ["price_list[0]", "price_list[1]", "price_list[2]", "price_list[3]", "price_list[4]", "price_list[5]", "price_list[6]", "price_list[7]", "price_list[8]", "price_list[9]", "price_list[10]", "price_list[11]", "price_list[12]", "price_list[13]", "price_list[14]", "price_list[15]", "price_list[16]", "price_list[17]", "price_list[18]", "price_list[19]", "price_list[20]", "price_list[21]", "price_list[22]", "price_list[23]", "price_list[24]", "price_list[25]", "price_list[26]", "price_list[27]", "price_list[28]", "price_list[29]", "price_list[30]", "price_list[31]", "price_list[32]", "price_list[33]", "price_list[34]", "price_list[35]", "price_list[36]", "price_list[37]", "price_list[38]", "price_list[39]", "price_list[40]", "price_list[41]", "price_list[42]", "price_list[43]", "price_list[44]", "price_list[45]", "price_list[46]", "price_list[47]", "price_list[48]", "price_list[49]"]
        self.cat_list = [self.cat_fea_list, self.cat_seq_fea]
        self.dense_list = [self.dense_fea, self.dense_seq_fea]

    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas
        self.cat_fea, self.cat_seq_fea = self.cat_fea_split
        self.cat_fea_emb, self.cat_seq_fea_emb = self.ptable_lookup(list_ids=[self.cat_fea, self.cat_seq_fea], v_name=self.__class__.__name__)
        self.dense_fea_split = dense_feas
        self.dense_fea, self.dense_seq_fea = self.dense_fea_split
        
    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=self.dense_fea, training=self.is_training,
                                                                     name='num_fea_batch_norm2')
        output_dense = se_block(normalized_numerical_fea_col, 1, 'RerankDense', self.is_training, self.params['se_type'])

        cate_feat_gather = tf.reshape(self.cat_fea_emb, [-1, embed_dim * len(self.cat_fea_list)])
        output_cat = se_block(cate_feat_gather, embed_dim, 'Category', self.is_training, self.params['se_type'])

        output = tf.concat([output_dense, output_cat], -1)
        return output

    def my_dense(self, inputs, output_dims, name, activation=tf.nn.relu):
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

    def mmoe_unit(self, trace_scene_list, can_input, name):
        #
        #can_input : [B, 300, emb]
        #
        # can_input = my_dense(can_input, base_embed_dim, 'MMOE_export_input'+name) # [B, 300, 8]

        trace_scene_list_emb = self.ptable_lookup2(list_ids=trace_scene_list, v_name=self.__class__.__name__+'mmoe_unit_lookup_'+name)
        can_output = self.can_unit(can_input, trace_scene_list_emb) # [B, 300, 8]
        return can_output

    def mmoe_model(self, model_input, scene_input, share_export_num, specific_export_num, name):  
        with tf.variable_scope('mmoe_model' + name, reuse=tf.AUTO_REUSE):
            # 1. export
            can_input = self.my_dense(model_input, self.base_embed_dim, 'MMOE_export_input') # [B, 300, 8]                    
            export_output = [self.mmoe_unit(scene_input + 10*i, can_input, 'MMOE_spec_export_%d'%i) for i in range(specific_export_num)] # [B, 300, 8]

            for i in range(share_export_num):
                e_o_1 = self.my_dense(model_input, self.base_embed_dim, 'MMOE_export_%d_1'%i) # [B, 300, 8]
                e_o_2 = self.my_dense(e_o_1, self.base_embed_dim, 'MMOE_export_%d_2'%i) # [B, 300, 8]
                export_output.append(e_o_2)
            export_output = tf.stack(export_output, -1) # [B, 300, emb, 4+4]
            # 2. gate, this is not MMOE, but is one-gate MOE
            gate_att = self.my_dense(model_input, share_export_num + specific_export_num, 'MMOE_gate')
            gate_att = tf.nn.softmax(gate_att, -1)
            gate_output = tf.reduce_mean(export_output * gate_att[:, :, None, :], -1) # [B, 300, emb]
            # 3. mlp tower
            can_output = self.mmoe_unit(scene_input + 10 * specific_export_num, gate_output, 'MMOE_mlp') # [B, 300, 8]
            return can_output
    
    def get_scene_params(self, scene_id_list, slot, share_export_num, specific_export_num, name):
        with tf.variable_scope('get_scene_params' + name, reuse=tf.AUTO_REUSE):
            # share_tower = [self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + i, v_name=self.__class__.__name__+'mmoe_unit_lookup_'+name) for i in range(share_export_num)]
            # specific_tower = [self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + share_export_num + 10 * i, v_name=self.__class__.__name__+'mmoe_unit_lookup_'+name) for i in range(specific_export_num)]
            share_tower = self.ptable_lookup2(list_ids=tf.zeros_like(scene_id_list) + (slot << 44), v_name=self.__class__.__name__+'lookup_share_tower')[0]
            specific_tower = self.ptable_lookup2(list_ids=scene_id_list + (slot << 44) + 1, v_name=self.__class__.__name__+'lookup_specific_tower')[0]
            return share_tower + specific_tower
            