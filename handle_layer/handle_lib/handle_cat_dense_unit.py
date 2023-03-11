#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import math

from handle_base import InputBase


class Dense(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(Dense, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.dense_feat_list = ["distance", "user_tag_pref_click_3day", "user_tag_pref_click_15day", "user_tag_pref_click_30day", "user_tag_pref_order_30day", "uv_cvr_15day", "pv_ctr_7day",
		                        "pv_ctr_15day", "pv_ctr_30day", "comment_uv", "comment_5star", "month_original_price", "distance_30day", "order_cnt_22hr", "food_num", "poi_person_avg",
		                        "poi_price_25per", "poi_price_100per", "user_price_50per", "user_price_75per", "user_poi_click_decay", "user_poi_order_decay", "view_poi_score_timedecay",
		                        "user_poi_ctr_timedecay", "user_poi_cvr_timedecay", "user_poi_cxr_timedecay", "view_cnt_after_last_clk", "day_since_last_click", "click_num_3day", "submit_num_3day",
		                        "click_num_7day", "submit_num_7day", "order_num_7day", "order_num_15day", "click_num_30day", "submit_num_30day", "order_num_30day", "user_30days_avg_discount_rate",
		                        "user_7days_click_discount_rate", "poi_7days_avg_discount_rate", "discount_rate_all_customer_7day", "poi_7days_avg_original_price", "user_view_poi_unique_num",
		                        "user_click_poi_unique_num", "user_view_poi_total_num", "user_click_poi_total_num", "user_view_trace_poi_avg_num", "user_click_trace_poi_avg_num",
		                        "user_order_trace_poi_avg_num", "user_view_trace_num", "user_click_trace_num", "user_order_trace_num", "user_trace_cxr", "user_unique_ctr", "user_unique_cvr",
		                        "user_unique_cxr", "user_total_ctr", "user_total_cxr", "user_click_concentre_ratio", "last_click_poi_time_interval_avg", "last_click_poi_time_interval_max",
		                        "last_order_uuid_time_interval_avg", "user_30_cnt_ad", "user_90_cnt_ad", "user_id_ord_poi_num", "user_id_dp_high_ord_num", "user_id_dp_low_ord_num",
		                        "user_id_total_ord_num_180day", "user_id_max_poi_order", "user_id_avg_poi_order", "user_id_dp_high_quality_poi_click_rate", "user_id_ka_click_rate",
		                        "user_id_ka_ord_rate", "sub_ord_num_7days", "sub_ord_num_90days", "sub_ord_num_weekdays", "sub_ord_num_total", "sub_ord_amt_30days", "sub_ord_amt_90days",
		                        "sub_ord_amt_weekdays", "sub_ord_amt_weekends", "user_order_poi_total_num", "user_order_avg_view_poi_num", "user_order_avg_click_poi_num", "coec_ctr_14", "coec_ctr_30",
		                        "coec_ctr_60", "online_delivery_fee", "onlne_original_delivery_fee", "online_delivery_time", "online_distance", "last_query_timediff",
		                        "uuid_poi_click_map_1day_idr_1116", "uuid_poi_click_map_12hour_idr_1116", "uuid_poi_click_map_1hour_idr_1116", "uuid_poi_order_map_1day_idr_1116",
		                        "uuid_poi_order_map_12hour_idr_1116", "uuid_poi_order_map_1hour_idr_1116", "poi_click_cnt_1day_idr_1110", "poi_click_cnt_12hour_idr_1110",
		                        "poi_click_cnt_1hour_idr_1110", "poi_search_click_count_15min_idr_1110", "poi_search_click_count_1hour_idr_1110", "poi_search_click_count_1day_idr_1110",
		                        "poi_search_submit_order_cnt_15minu_rt_1110", "poi_search_submit_order_cnt_1hour_rt_1110", "poi_search_submit_order_cnt_1day_rt_1110", "user_poi_total_stay_time_1day",
		                        "user_poi_avg_stay_time_1day", "user_poi_view_cnt_1day", "user_poi_total_stay_time_3day", "user_poi_avg_stay_time_3day", "user_poi_view_cnt_3day",
		                        "user_poi_total_stay_time_7day", "user_poi_avg_stay_time_7day", "user_poi_view_cnt_7day", "user_poi_total_stay_time_15day", "user_poi_avg_stay_time_15day",
		                        "user_poi_view_cnt_15day", "user_poi_total_stay_time_30day", "user_poi_avg_stay_time_30day", "user_poi_view_cnt_30day", "user_poi_total_stay_time_90day",
		                        "user_poi_avg_stay_time_90day", "user_poi_view_cnt_90day", "poi_hour_expose_num_28day", "poi_hour_click_num_28day", "poi_hour_order_num_28day",
		                        "poi_hour_total_price_28day", "poi_hour_ctr_28day", "poi_hour_expose_num_7day", "poi_hour_click_num_7day", "poi_hour_order_num_7day", "poi_hour_total_price_7day",
		                        "poi_hour_ctr_7day", "last_buy_days", "cnt_90", "cnt_180", "browse_slient", "member_7_pv", "member_30_pv", "member_60_pv", "member_90_pv", "member_incr_coupon_cnt",
		                        "member_incr_coupon_freeze_cnt", "member_coupon_cnt", "member_coupon_freeze_cnt", "user_large_amount_coup_days", "is_receive_red_7days", "user_grand_total_vp_ord_num",
		                        "user_is_bind_wm_user", "user_unuse_mt_vip_cpn_num", "user_mt_vip_cpn_valid_left_days", "user_meituanapp", "user_dazhongapp", "user_zhifubao", "user_shoujitaobao",
		                        "user_meituanwaimai", "user_koubei", "user_eleme", "poi_wm_level", "poi_mt_score", "poi_dp_score", "poi_wm_poi_score",
		                        "poi_ord_num_last1_7day_avg_second_city_normalization_1", "poi_actual_price_last1_7day_avg_second_city_normalization_1",
		                        "poi_exposure_cnt_last1_7day_sum_second_city_normalization_1", "poi_visit_cnt_last1_7day_sum_second_city_normalization_1", "poi_breakfirst_ord_num_30d_rate",
		                        "poi_lunch_ord_num_30d_rate", "poi_dinner_ord_num_30d_rate", "poi_supper_ord_num_30d_rate", "poi_food_good_comment", "poi_food_bad_comment",
		                        "poi_food_have_comment_num", "uuid_cart_act_cnt_1day", "uuid_cart_act_cnt_7day", "uuid_cart_act_cnt_30day", "uuid_cart_spu_cnt_1day", "uuid_cart_spu_cnt_7day",
		                        "uuid_cart_spu_cnt_30day", "poi_cart_act_cnt_1day", "poi_cart_act_cnt_7day", "poi_cart_act_cnt_30day", "poi_cart_spu_cnt_1day", "poi_cart_spu_cnt_7day",
		                        "poi_cart_spu_cnt_30day", "expose_cnt_1day", "expose_cnt_3day", "expose_cnt_7day", "expose_cnt_15day", "expose_cnt_30day", "click_cnt_1day", "click_cnt_3day",
		                        "click_cnt_7day", "click_cnt_15day", "click_cnt_30day", "submit_cnt_3day", "submit_cnt_7day", "submit_cnt_30day", "click_ratio_1day", "click_ratio_3day",
		                        "click_ratio_7day", "click_ratio_15day", "click_ratio_30day", "submit_expose_ratio_3day", "submit_expose_ratio_7day", "submit_expose_ratio_30day",
		                        "submit_click_ratio_3day", "submit_click_ratio_7day", "submit_click_ratio_30day", "last_click_poi_time_interval[0]", "last_click_poi_time_interval[1]",
		                        "last_click_poi_time_interval[2]", "last_click_poi_time_interval[3]"]

		self.gather_feas.append_dense_feas([self.dense_feat_list])
		# self.gather_feas.append_cat_feas([self])
		# self.gather_feas_dict.dense.add(self.dense_feat_list)
		self.dense_list = [self.dense_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		dense_feat_size = len(self.dense_feat_list)

		# if dense_feat_size == 0 or self.dense_feat_list[0] == '*':
		#     dense_feat = dense_features
		# else:
		#     feat_index = index_of_tensor(self.dense_columns_info.index_of_column, self.dense_feat_list)
		#     dense_feat = tf.gather(dense_features, feat_index, axis=1)
		dense_feat = self.dense_fea_split

		zeros_tensor = tf.zeros([1, dense_feat_size], tf.float32)
		imputation_w1 = tf.get_variable('imputation_w1', shape=[1, dense_feat_size],
		                                initializer=tf.random_normal_initializer())
		imputation_w2 = tf.get_variable('imputation_w2', shape=[1, dense_feat_size],
		                                initializer=tf.random_normal_initializer())

		dense_fea_col = tf.multiply(tf.cast(tf.math.equal(dense_feat, zeros_tensor), tf.float32),
		                            imputation_w1) + \
		                tf.multiply(tf.where(tf.math.equal(dense_feat, zeros_tensor),
		                                     tf.zeros_like(dense_feat), dense_feat),
		                            imputation_w2)
		normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=dense_fea_col, training=self.is_training,
		                                                             name='num_fea_batch_norm')
		normed_dense_feat = tf.nn.tanh(normalized_numerical_fea_col)

		return se_block(normed_dense_feat, 1, 'Dense', self.is_training, self.params['se_type'])


class RerankDense(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(RerankDense, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.rerank_feat_list = ["rerank_top40_ctr_list[0]", "rerank_top40_ctr_list[1]", "rerank_top40_ctr_list[2]", "rerank_top40_ctr_list[3]", "rerank_top40_ctr_list[4]", "rerank_top40_ctr_list[5]",
		                         "rerank_top40_ctr_list[6]", "rerank_top40_ctr_list[7]", "rerank_top40_ctr_list[8]", "rerank_top40_ctr_list[9]", "rerank_top40_ctr_list[10]",
		                         "rerank_top40_ctr_list[11]", "rerank_top40_ctr_list[12]", "rerank_top40_ctr_list[13]", "rerank_top40_ctr_list[14]", "rerank_top40_ctr_list[15]",
		                         "rerank_top40_ctr_list[16]", "rerank_top40_ctr_list[17]", "rerank_top40_ctr_list[18]", "rerank_top40_ctr_list[19]", "rerank_top40_ctr_list[20]",
		                         "rerank_top40_ctr_list[21]", "rerank_top40_ctr_list[22]", "rerank_top40_ctr_list[23]", "rerank_top40_ctr_list[24]", "rerank_top40_ctr_list[25]",
		                         "rerank_top40_ctr_list[26]", "rerank_top40_ctr_list[27]", "rerank_top40_ctr_list[28]", "rerank_top40_ctr_list[29]", "rerank_top40_ctr_list[30]",
		                         "rerank_top40_ctr_list[31]", "rerank_top40_ctr_list[32]", "rerank_top40_ctr_list[33]", "rerank_top40_ctr_list[34]", "rerank_top40_ctr_list[35]",
		                         "rerank_top40_ctr_list[36]", "rerank_top40_ctr_list[37]", "rerank_top40_ctr_list[38]", "rerank_top40_ctr_list[39]", "rerank_top40_cvr_list[0]",
		                         "rerank_top40_cvr_list[1]", "rerank_top40_cvr_list[2]", "rerank_top40_cvr_list[3]", "rerank_top40_cvr_list[4]", "rerank_top40_cvr_list[5]", "rerank_top40_cvr_list[6]",
		                         "rerank_top40_cvr_list[7]", "rerank_top40_cvr_list[8]", "rerank_top40_cvr_list[9]", "rerank_top40_cvr_list[10]", "rerank_top40_cvr_list[11]",
		                         "rerank_top40_cvr_list[12]", "rerank_top40_cvr_list[13]", "rerank_top40_cvr_list[14]", "rerank_top40_cvr_list[15]", "rerank_top40_cvr_list[16]",
		                         "rerank_top40_cvr_list[17]", "rerank_top40_cvr_list[18]", "rerank_top40_cvr_list[19]", "rerank_top40_cvr_list[20]", "rerank_top40_cvr_list[21]",
		                         "rerank_top40_cvr_list[22]", "rerank_top40_cvr_list[23]", "rerank_top40_cvr_list[24]", "rerank_top40_cvr_list[25]", "rerank_top40_cvr_list[26]",
		                         "rerank_top40_cvr_list[27]", "rerank_top40_cvr_list[28]", "rerank_top40_cvr_list[29]", "rerank_top40_cvr_list[30]", "rerank_top40_cvr_list[31]",
		                         "rerank_top40_cvr_list[32]", "rerank_top40_cvr_list[33]", "rerank_top40_cvr_list[34]", "rerank_top40_cvr_list[35]", "rerank_top40_cvr_list[36]",
		                         "rerank_top40_cvr_list[37]", "rerank_top40_cvr_list[38]", "rerank_top40_cvr_list[39]", "rerank_top40_cxr_list[0]", "rerank_top40_cxr_list[1]",
		                         "rerank_top40_cxr_list[2]", "rerank_top40_cxr_list[3]", "rerank_top40_cxr_list[4]", "rerank_top40_cxr_list[5]", "rerank_top40_cxr_list[6]", "rerank_top40_cxr_list[7]",
		                         "rerank_top40_cxr_list[8]", "rerank_top40_cxr_list[9]", "rerank_top40_cxr_list[10]", "rerank_top40_cxr_list[11]", "rerank_top40_cxr_list[12]",
		                         "rerank_top40_cxr_list[13]", "rerank_top40_cxr_list[14]", "rerank_top40_cxr_list[15]", "rerank_top40_cxr_list[16]", "rerank_top40_cxr_list[17]",
		                         "rerank_top40_cxr_list[18]", "rerank_top40_cxr_list[19]", "rerank_top40_cxr_list[20]", "rerank_top40_cxr_list[21]", "rerank_top40_cxr_list[22]",
		                         "rerank_top40_cxr_list[23]", "rerank_top40_cxr_list[24]", "rerank_top40_cxr_list[25]", "rerank_top40_cxr_list[26]", "rerank_top40_cxr_list[27]",
		                         "rerank_top40_cxr_list[28]", "rerank_top40_cxr_list[29]", "rerank_top40_cxr_list[30]", "rerank_top40_cxr_list[31]", "rerank_top40_cxr_list[32]",
		                         "rerank_top40_cxr_list[33]", "rerank_top40_cxr_list[34]", "rerank_top40_cxr_list[35]", "rerank_top40_cxr_list[36]", "rerank_top40_cxr_list[37]",
		                         "rerank_top40_cxr_list[38]", "rerank_top40_cxr_list[39]", "rerank_top40_vs_list[0]", "rerank_top40_vs_list[1]", "rerank_top40_vs_list[2]", "rerank_top40_vs_list[3]",
		                         "rerank_top40_vs_list[4]", "rerank_top40_vs_list[5]", "rerank_top40_vs_list[6]", "rerank_top40_vs_list[7]", "rerank_top40_vs_list[8]", "rerank_top40_vs_list[9]",
		                         "rerank_top40_vs_list[10]", "rerank_top40_vs_list[11]", "rerank_top40_vs_list[12]", "rerank_top40_vs_list[13]", "rerank_top40_vs_list[14]", "rerank_top40_vs_list[15]",
		                         "rerank_top40_vs_list[16]", "rerank_top40_vs_list[17]", "rerank_top40_vs_list[18]", "rerank_top40_vs_list[19]", "rerank_top40_vs_list[20]", "rerank_top40_vs_list[21]",
		                         "rerank_top40_vs_list[22]", "rerank_top40_vs_list[23]", "rerank_top40_vs_list[24]", "rerank_top40_vs_list[25]", "rerank_top40_vs_list[26]", "rerank_top40_vs_list[27]",
		                         "rerank_top40_vs_list[28]", "rerank_top40_vs_list[29]", "rerank_top40_vs_list[30]", "rerank_top40_vs_list[31]", "rerank_top40_vs_list[32]", "rerank_top40_vs_list[33]",
		                         "rerank_top40_vs_list[34]", "rerank_top40_vs_list[35]", "rerank_top40_vs_list[36]", "rerank_top40_vs_list[37]", "rerank_top40_vs_list[38]", "rerank_top40_vs_list[39]",
		                         "rerank_top40_cs_list[0]", "rerank_top40_cs_list[1]", "rerank_top40_cs_list[2]", "rerank_top40_cs_list[3]", "rerank_top40_cs_list[4]", "rerank_top40_cs_list[5]",
		                         "rerank_top40_cs_list[6]", "rerank_top40_cs_list[7]", "rerank_top40_cs_list[8]", "rerank_top40_cs_list[9]", "rerank_top40_cs_list[10]", "rerank_top40_cs_list[11]",
		                         "rerank_top40_cs_list[12]", "rerank_top40_cs_list[13]", "rerank_top40_cs_list[14]", "rerank_top40_cs_list[15]", "rerank_top40_cs_list[16]", "rerank_top40_cs_list[17]",
		                         "rerank_top40_cs_list[18]", "rerank_top40_cs_list[19]", "rerank_top40_cs_list[20]", "rerank_top40_cs_list[21]", "rerank_top40_cs_list[22]", "rerank_top40_cs_list[23]",
		                         "rerank_top40_cs_list[24]", "rerank_top40_cs_list[25]", "rerank_top40_cs_list[26]", "rerank_top40_cs_list[27]", "rerank_top40_cs_list[28]", "rerank_top40_cs_list[29]",
		                         "rerank_top40_cs_list[30]", "rerank_top40_cs_list[31]", "rerank_top40_cs_list[32]", "rerank_top40_cs_list[33]", "rerank_top40_cs_list[34]", "rerank_top40_cs_list[35]",
		                         "rerank_top40_cs_list[36]", "rerank_top40_cs_list[37]", "rerank_top40_cs_list[38]", "rerank_top40_cs_list[39]"]
		self.gather_feas.append_dense_feas([self.rerank_feat_list])
		self.dense_list = [self.rerank_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		dense_feat = self.dense_fea_split
		normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=dense_feat, training=self.is_training,
		                                                             name='num_fea_batch_norm2')

		return se_block(normalized_numerical_fea_col, 1, 'RerankDense', self.is_training, self.params['se_type'])


class LASTDense(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(LASTDense, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.other_dense = [u'ec_post_req_ad_poi_ctr_list[0]', u'ec_post_req_ad_poi_ctr_list[10]', u'ec_post_req_ad_poi_ctr_list[11]', u'ec_post_req_ad_poi_ctr_list[12]',
		                    u'ec_post_req_ad_poi_ctr_list[13]', u'ec_post_req_ad_poi_ctr_list[14]', u'ec_post_req_ad_poi_ctr_list[15]', u'ec_post_req_ad_poi_ctr_list[16]',
		                    u'ec_post_req_ad_poi_ctr_list[17]', u'ec_post_req_ad_poi_ctr_list[18]', u'ec_post_req_ad_poi_ctr_list[19]', u'ec_post_req_ad_poi_ctr_list[1]',
		                    u'ec_post_req_ad_poi_ctr_list[20]', u'ec_post_req_ad_poi_ctr_list[21]', u'ec_post_req_ad_poi_ctr_list[22]', u'ec_post_req_ad_poi_ctr_list[23]',
		                    u'ec_post_req_ad_poi_ctr_list[24]', u'ec_post_req_ad_poi_ctr_list[2]', u'ec_post_req_ad_poi_ctr_list[3]', u'ec_post_req_ad_poi_ctr_list[4]',
		                    u'ec_post_req_ad_poi_ctr_list[5]', u'ec_post_req_ad_poi_ctr_list[6]', u'ec_post_req_ad_poi_ctr_list[7]', u'ec_post_req_ad_poi_ctr_list[8]',
		                    u'ec_post_req_ad_poi_ctr_list[9]', u'ec_post_req_ad_poi_cvr_list[0]', u'ec_post_req_ad_poi_cvr_list[10]', u'ec_post_req_ad_poi_cvr_list[11]',
		                    u'ec_post_req_ad_poi_cvr_list[12]', u'ec_post_req_ad_poi_cvr_list[13]', u'ec_post_req_ad_poi_cvr_list[14]', u'ec_post_req_ad_poi_cvr_list[15]',
		                    u'ec_post_req_ad_poi_cvr_list[16]', u'ec_post_req_ad_poi_cvr_list[17]', u'ec_post_req_ad_poi_cvr_list[18]', u'ec_post_req_ad_poi_cvr_list[19]',
		                    u'ec_post_req_ad_poi_cvr_list[1]', u'ec_post_req_ad_poi_cvr_list[20]', u'ec_post_req_ad_poi_cvr_list[21]', u'ec_post_req_ad_poi_cvr_list[22]',
		                    u'ec_post_req_ad_poi_cvr_list[23]', u'ec_post_req_ad_poi_cvr_list[24]', u'ec_post_req_ad_poi_cvr_list[2]', u'ec_post_req_ad_poi_cvr_list[3]',
		                    u'ec_post_req_ad_poi_cvr_list[4]', u'ec_post_req_ad_poi_cvr_list[5]', u'ec_post_req_ad_poi_cvr_list[6]', u'ec_post_req_ad_poi_cvr_list[7]',
		                    u'ec_post_req_ad_poi_cvr_list[8]', u'ec_post_req_ad_poi_cvr_list[9]', u'ec_post_req_ad_poi_gmv_list[0]', u'ec_post_req_ad_poi_gmv_list[10]',
		                    u'ec_post_req_ad_poi_gmv_list[11]', u'ec_post_req_ad_poi_gmv_list[12]', u'ec_post_req_ad_poi_gmv_list[13]', u'ec_post_req_ad_poi_gmv_list[14]',
		                    u'ec_post_req_ad_poi_gmv_list[15]', u'ec_post_req_ad_poi_gmv_list[16]', u'ec_post_req_ad_poi_gmv_list[17]', u'ec_post_req_ad_poi_gmv_list[18]',
		                    u'ec_post_req_ad_poi_gmv_list[19]', u'ec_post_req_ad_poi_gmv_list[1]', u'ec_post_req_ad_poi_gmv_list[20]', u'ec_post_req_ad_poi_gmv_list[21]',
		                    u'ec_post_req_ad_poi_gmv_list[22]', u'ec_post_req_ad_poi_gmv_list[23]', u'ec_post_req_ad_poi_gmv_list[24]', u'ec_post_req_ad_poi_gmv_list[2]',
		                    u'ec_post_req_ad_poi_gmv_list[3]', u'ec_post_req_ad_poi_gmv_list[4]', u'ec_post_req_ad_poi_gmv_list[5]', u'ec_post_req_ad_poi_gmv_list[6]',
		                    u'ec_post_req_ad_poi_gmv_list[7]', u'ec_post_req_ad_poi_gmv_list[8]', u'ec_post_req_ad_poi_gmv_list[9]', u'side_debias_model_score_list[0]',
		                    u'side_debias_model_score_list[10]', u'side_debias_model_score_list[11]', u'side_debias_model_score_list[12]', u'side_debias_model_score_list[13]',
		                    u'side_debias_model_score_list[14]', u'side_debias_model_score_list[15]', u'side_debias_model_score_list[16]', u'side_debias_model_score_list[17]',
		                    u'side_debias_model_score_list[18]', u'side_debias_model_score_list[19]', u'side_debias_model_score_list[1]', u'side_debias_model_score_list[20]',
		                    u'side_debias_model_score_list[21]', u'side_debias_model_score_list[22]', u'side_debias_model_score_list[23]', u'side_debias_model_score_list[24]',
		                    u'side_debias_model_score_list[2]', u'side_debias_model_score_list[3]', u'side_debias_model_score_list[4]', u'side_debias_model_score_list[5]',
		                    u'side_debias_model_score_list[6]', u'side_debias_model_score_list[7]', u'side_debias_model_score_list[8]', u'side_debias_model_score_list[9]', u'uuid_cluster_center_1[0]',
		                    u'uuid_cluster_center_1[10]', u'uuid_cluster_center_1[11]', u'uuid_cluster_center_1[12]', u'uuid_cluster_center_1[13]', u'uuid_cluster_center_1[14]',
		                    u'uuid_cluster_center_1[15]', u'uuid_cluster_center_1[1]', u'uuid_cluster_center_1[2]', u'uuid_cluster_center_1[3]', u'uuid_cluster_center_1[4]',
		                    u'uuid_cluster_center_1[5]', u'uuid_cluster_center_1[6]', u'uuid_cluster_center_1[7]', u'uuid_cluster_center_1[8]', u'uuid_cluster_center_1[9]',
		                    u'uuid_cluster_center_2[0]', u'uuid_cluster_center_2[10]', u'uuid_cluster_center_2[11]', u'uuid_cluster_center_2[12]', u'uuid_cluster_center_2[13]',
		                    u'uuid_cluster_center_2[14]', u'uuid_cluster_center_2[15]', u'uuid_cluster_center_2[1]', u'uuid_cluster_center_2[2]', u'uuid_cluster_center_2[3]',
		                    u'uuid_cluster_center_2[4]', u'uuid_cluster_center_2[5]', u'uuid_cluster_center_2[6]', u'uuid_cluster_center_2[7]', u'uuid_cluster_center_2[8]',
		                    u'uuid_cluster_center_2[9]', u'uuid_cluster_center_3[0]', u'uuid_cluster_center_3[10]', u'uuid_cluster_center_3[11]', u'uuid_cluster_center_3[12]',
		                    u'uuid_cluster_center_3[13]', u'uuid_cluster_center_3[14]', u'uuid_cluster_center_3[15]', u'uuid_cluster_center_3[1]', u'uuid_cluster_center_3[2]',
		                    u'uuid_cluster_center_3[3]', u'uuid_cluster_center_3[4]', u'uuid_cluster_center_3[5]', u'uuid_cluster_center_3[6]', u'uuid_cluster_center_3[7]',
		                    u'uuid_cluster_center_3[8]', u'uuid_cluster_center_3[9]', u'uuid_poi_click_list_embedding[0]', u'uuid_poi_click_list_embedding[10]', u'uuid_poi_click_list_embedding[11]',
		                    u'uuid_poi_click_list_embedding[12]', u'uuid_poi_click_list_embedding[13]', u'uuid_poi_click_list_embedding[14]', u'uuid_poi_click_list_embedding[15]',
		                    u'uuid_poi_click_list_embedding[1]', u'uuid_poi_click_list_embedding[2]', u'uuid_poi_click_list_embedding[3]', u'uuid_poi_click_list_embedding[4]',
		                    u'uuid_poi_click_list_embedding[5]', u'uuid_poi_click_list_embedding[6]', u'uuid_poi_click_list_embedding[7]', u'uuid_poi_click_list_embedding[8]',
		                    u'uuid_poi_click_list_embedding[9]']
		self.gather_feas.append_dense_feas([self.other_dense])
		self.dense_list = [self.other_dense]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		dense_feat = self.dense_fea_split
		normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=dense_feat, training=self.is_training,
		                                                             name='num_fea_batch_norm3')

		return se_block(normalized_numerical_fea_col, 1, 'otherDense', self.is_training, self.params['se_type'])


class Category(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(Category, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.cate_feat_list = ["discrete_slot_int64", "discrete_poi_position_int64", "discrete_client_type_int64", "discrete_app_version_int64", "discrete_hour_of_day_int64",
		                       "discrete_week_day_int64", "discrete_city_id_int64", "discrete_device_type_int64", "discrete_name_int64", "discrete_pic_url_int64", "high_confidence_gender_int64",
		                       "high_confidence_age_int64", "discrete_career_int64", "discrete_is_super_poi_int64", "discrete_consume_style_int64", "discrete_sensitivity_level_int64",
		                       "discrete_clk_third_tag_2month_ad_1_int64", "discrete_clk_third_tag_2month_ad_2_int64", "discrete_clk_third_tag_2month_ad_3_int64",
		                       "discrete_clk_third_tag_2month_ad_4_int64", "discrete_clk_third_tag_2month_ad_5_int64", "discrete_clk_third_tag_2month_ad_6_int64",
		                       "discrete_sub_third_tag_2month_ad_1_int64", "discrete_sub_third_tag_2month_ad_2_int64", "discrete_sub_third_tag_2month_ad_3_int64",
		                       "discrete_sub_third_tag_2month_ad_4_int64", "discrete_sub_third_tag_2month_ad_5_int64", "discrete_sub_third_tag_2month_ad_6_int64",
		                       "discrete_clk_cluster_2month_ad_1_int64", "discrete_clk_cluster_2month_ad_2_int64", "discrete_clk_cluster_2month_ad_3_int64", "discrete_clk_cluster_2month_ad_4_int64",
		                       "discrete_clk_cluster_2month_ad_5_int64", "discrete_clk_cluster_2month_ad_6_int64", "discrete_sub_cluster_2month_ad_1_int64", "discrete_sub_cluster_2month_ad_2_int64",
		                       "discrete_sub_cluster_2month_ad_3_int64", "discrete_sub_cluster_2month_ad_4_int64", "discrete_sub_cluster_2month_ad_5_int64", "discrete_sub_cluster_2month_ad_6_int64",
		                       "online_delivery_type_int64", "online_recommend_int64", "online_logo_tag_int64", "last_query_int64", "last_query_first_tag_int64", "last_query_second_tag_int64",
		                       "last_query_third_tag_int64", "discrete_post_clk_is_coupon_clked_int64", "discrete_post_clk_is_redpacket_clked_int64",
		                       "discrete_post_clk_is_activity_detail_viewed_int64", "discrete_post_clk_duration_int64", "discrete_post_clk_expose_spu_num_int64", "discrete_post_clk_clk_spu_num_int64",
		                       "discrete_post_clk_is_comment_tab_viewed_int64", "discrete_post_clk_is_poi_tab_viewed_int64", "discrete_post_clk_is_poi_contacted_int64",
		                       "discrete_post_clk_clk_spu_list_int64[0]", "discrete_post_clk_in_cart_spu_list_int64[0]", "discrete_post_clk_query_list_int64[0]"]
		self.gather_feas.append_cat_feas([self.cate_feat_list])
		self.cat_list = [self.cate_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)[0]  # for i, feat_idx in enumerate(self.cat_fea_split)]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		# feat_index = index_of_tensor(self.cat_columns_info.index_of_column, self.cate_feat_list)

		# cate_feat_gather = tf.gather(cat_features_embed, feat_index, axis=2)
		cate_feat_gather = tf.reshape(self.cat_fea_emb, [-1, embed_dim * len(self.cate_feat_list)])
		return se_block(cate_feat_gather, embed_dim, 'Category', self.is_training, self.params['se_type'])


class CategoryKws(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(CategoryKws, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.poi_name_kws_feat_list = ["mmhash_poi_name_kws_id_list_int64[%d]" % i for i in range(9)]
		self.gather_feas.append_cat_feas([self.poi_name_kws_feat_list])
		self.cat_list = [self.poi_name_kws_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		cate_kws_feat = tf.reshape(self.cat_fea_emb, [-1, embed_dim * len(self.poi_name_kws_feat_list)])
		se_feat = se_block(cate_kws_feat, embed_dim, 'CategoryKws', self.is_training, self.params['se_type'])
		return se_feat


class CategoryMmhashSlimv2(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(CategoryMmhashSlimv2, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.mmhash_slimv2_feat_list = ["d_user_tag_pref_click_3day_int64", "d_user_tag_pref_click_15day_int64", "d_user_tag_pref_click_30day_int64", "d_user_tag_pref_order_30day_int64",
		                                "d_comment_5star_int64", "d_month_original_price_int64", "d_food_num_int64", "d_poi_price_25per_int64", "d_user_price_50per_int64",
		                                "d_user_30days_avg_discount_rate_int64",
		                                "d_user_7days_click_discount_rate_int64", "d_poi_7days_avg_discount_rate_int64", "d_user_view_poi_unique_num_int64", "d_user_total_ctr_int64",
		                                "d_user_total_cxr_int64",
		                                "d_user_30_cnt_ad_int64", "d_user_id_dp_high_ord_num_int64", "d_user_id_dp_low_ord_num_int64", "d_user_id_ka_click_rate_int64", "d_user_id_ka_ord_rate_int64",
		                                "d_sub_ord_num_weekdays_int64", "d_sub_ord_amt_weekdays_int64", "d_coec_ctr_14_int64", "d_coec_ctr_30_int64", "d_coec_ctr_60_int64",
		                                "mmhash_poi_cluster_id_int64",
		                                "mmhash_poi_clk_top_dpc1_int64", "mmhash_poi_clk_top_dpc2_int64", "mmhash_poi_clk_top_dpc3_int64", "mmhash_poi_sub_top_dpc1_int64",
		                                "mmhash_poi_sub_top_dpc2_int64",
		                                "mmhash_poi_sub_top_dpc3_int64", "mmhash_user_clk_cluster_1_int64", "mmhash_user_clk_cluster_2_int64", "mmhash_user_clk_cluster_3_int64",
		                                "mmhash_user_clk_cluster_4_int64",
		                                "mmhash_user_clk_cluster_5_int64", "mmhash_user_clk_cluster_6_int64", "mmhash_user_sub_cluster_1_int64", "mmhash_user_sub_cluster_2_int64",
		                                "mmhash_user_sub_cluster_3_int64",
		                                "mmhash_user_food_cluster_1_int64", "mmhash_user_food_cluster_2_int64", "mmhash_user_food_cluster_3_int64", "mmhash_user_clk_third_tag1_int64",
		                                "mmhash_user_clk_third_tag2_int64",
		                                "mmhash_user_clk_third_tag3_int64", "mmhash_user_clk_third_tag4_int64", "mmhash_user_clk_third_tag5_int64", "mmhash_user_clk_third_tag6_int64",
		                                "mmhash_user_sub_third_tag1_int64",
		                                "mmhash_user_sub_third_tag2_int64", "mmhash_user_sub_third_tag3_int64", "mmhash_user_stem_name1_int64", "mmhash_user_stem_name2_int64",
		                                "mmhash_user_stem_name3_int64",
		                                "mmhash_user_clk_dpc1_int64", "mmhash_user_clk_dpc2_int64", "mmhash_user_clk_dpc3_int64", "mmhash_user_sub_dpc1_int64", "mmhash_user_sub_dpc2_int64",
		                                "mmhash_user_sub_dpc3_int64",
		                                "mmhash_user_clk_top_poi1_int64", "mmhash_user_clk_top_poi2_int64", "mmhash_user_clk_top_poi3_int64", "mmhash_user_sub_top_poi1_int64",
		                                "mmhash_user_sub_top_poi2_int64",
		                                "mmhash_user_sub_top_poi3_int64", "mmhash_user_cluster_int64", "mmhash_most_90d_visit_aoi_id_int64", "mmhash_most_90d_visit_aoi_type_int64",
		                                "mmhash_most_90d_gmv_aoi_id_int64",
		                                "mmhash_most_90d_gmv_aoi_type_int64", "mmhash_end_aor_type_int64", "mmhash_end_poi_aor_id_int64", "mmhash_end_category_int64",
		                                "mmhash_address_category_id_int64", "mmhash_end_location_id_int64",
		                                "mmhash_nation_code_int64", "mmhash_geotag_id_work_int64", "mmhash_geotag_id_home_int64", "mmhash_usercls_third_tag_id_int64", "mmhash_usercls_poi_id_int64",
		                                "mmhash_usercls_poi_aor_id_int64",
		                                "mmhash_usercls_brand_ka_int64", "mmhash_usercls_hour_int64", "mmhash_usercls_week_int64", "mmhash_usercls_slot_int64",
		                                "mmhash_most_90d_visit_aoi_id_cross_int64",
		                                "mmhash_most_90d_gmv_aoi_id_cross_int64", "mmhash_end_poi_aor_id_cross_int64", "mmhash_address_category_id_cross_int64", "mmhash_geotag_id_work_cross_int64",
		                                "mmhash_area_id_work_cross_int64",
		                                "mmhash_geotag_id_home_cross_int64", "mmhash_area_id_home_cross_int64"]
		self.gather_feas.append_cat_feas([self.mmhash_slimv2_feat_list])
		self.cat_list = [self.mmhash_slimv2_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		cate_mmhash_slimv2_feat = tf.reshape(self.cat_fea_emb, [-1, embed_dim * len(self.mmhash_slimv2_feat_list)])
		se_feat = se_block(cate_mmhash_slimv2_feat, embed_dim, 'CategoryMmhashSlimv2', self.is_training, self.params['se_type'])
		return se_feat


class CategoryPooling(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(CategoryPooling, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.cate_feat_list = ["discrete_slot_int64", "discrete_poi_position_int64", "discrete_client_type_int64", "discrete_app_version_int64", "discrete_hour_of_day_int64",
		                       "discrete_week_day_int64", "discrete_city_id_int64", "discrete_device_type_int64", "discrete_name_int64", "discrete_pic_url_int64", "high_confidence_gender_int64",
		                       "high_confidence_age_int64", "discrete_career_int64", "discrete_is_super_poi_int64", "discrete_consume_style_int64", "discrete_sensitivity_level_int64",
		                       "discrete_clk_third_tag_2month_ad_1_int64", "discrete_clk_third_tag_2month_ad_2_int64", "discrete_clk_third_tag_2month_ad_3_int64",
		                       "discrete_clk_third_tag_2month_ad_4_int64", "discrete_clk_third_tag_2month_ad_5_int64", "discrete_clk_third_tag_2month_ad_6_int64",
		                       "discrete_sub_third_tag_2month_ad_1_int64", "discrete_sub_third_tag_2month_ad_2_int64", "discrete_sub_third_tag_2month_ad_3_int64",
		                       "discrete_sub_third_tag_2month_ad_4_int64", "discrete_sub_third_tag_2month_ad_5_int64", "discrete_sub_third_tag_2month_ad_6_int64",
		                       "discrete_clk_cluster_2month_ad_1_int64", "discrete_clk_cluster_2month_ad_2_int64", "discrete_clk_cluster_2month_ad_3_int64", "discrete_clk_cluster_2month_ad_4_int64",
		                       "discrete_clk_cluster_2month_ad_5_int64", "discrete_clk_cluster_2month_ad_6_int64", "discrete_sub_cluster_2month_ad_1_int64", "discrete_sub_cluster_2month_ad_2_int64",
		                       "discrete_sub_cluster_2month_ad_3_int64", "discrete_sub_cluster_2month_ad_4_int64", "discrete_sub_cluster_2month_ad_5_int64", "discrete_sub_cluster_2month_ad_6_int64",
		                       "online_delivery_type_int64", "online_recommend_int64", "online_logo_tag_int64", "last_query_int64", "last_query_first_tag_int64", "last_query_second_tag_int64",
		                       "last_query_third_tag_int64", "discrete_post_clk_is_coupon_clked_int64", "discrete_post_clk_is_redpacket_clked_int64",
		                       "discrete_post_clk_is_activity_detail_viewed_int64", "discrete_post_clk_duration_int64", "discrete_post_clk_expose_spu_num_int64", "discrete_post_clk_clk_spu_num_int64",
		                       "discrete_post_clk_is_comment_tab_viewed_int64", "discrete_post_clk_is_poi_tab_viewed_int64", "discrete_post_clk_is_poi_contacted_int64",
		                       "discrete_post_clk_clk_spu_list_int64[0]", "discrete_post_clk_in_cart_spu_list_int64[0]", "discrete_post_clk_query_list_int64[0]"]
		self.cat_list = [self.cate_feat_list]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)[0]  # for i, feat_idx in enumerate(self.cat_fea_split)]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		cate_feat_gather = tf.reshape(self.cat_fea_emb, [-1, embed_dim * len(self.cate_feat_list)])

		return se_block(cate_feat_gather, embed_dim, 'Category', self.is_training, self.params['se_type'])


class PoiTextEmbedDense(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(PoiTextEmbedDense, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.poi_text_embed_num = 4
		self.poi_text_embed_dim = 32
		self.poi_text_feat_list = [
			["list_poi_id_text_embedding_for_self_v2[%d]" % i for i in range(self.poi_text_embed_dim)],
			["list_poi_id_text_embedding_for_list_last_1st_click_v2[%d]" % i for i in range(self.poi_text_embed_dim)],
			["list_poi_id_text_embedding_for_list_last_2nd_click_v2[%d]" % i for i in range(self.poi_text_embed_dim)],
			["list_poi_id_text_embedding_for_list_last_3rd_click_v2[%d]" % i for i in range(self.poi_text_embed_dim)]]
		self.poi_text_feat = [v for l in self.poi_text_feat_list for v in l]
		self.gather_feas.append_dense_feas([self.poi_text_feat])
		self.dense_list = [self.poi_text_feat]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# dense_feas
		self.dense_fea_split = dense_feas[0]

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		""" fasttext 预训练text embedding, 已经归一化"""
		# feat_index = index_of_tensor(self.dense_columns_info.index_of_column, self.poi_text_feat)
		# pretrain_text_embed = tf.gather(dense_features, feat_index, axis=1)
		pretrain_text_embed = tf.reshape(self.dense_fea_split, [-1, self.poi_text_embed_num, self.poi_text_embed_dim])
		cur_poi_text_embed = tf.slice(pretrain_text_embed, [0, 0, 0], [-1, 1, -1])
		# cur_poi_text_embed [b, 1, dim]

		last_pois_text_embed = tf.slice(pretrain_text_embed, [0, 1, 0], [-1, -1, -1])
		# last_pois_text_embed [b, poi_text_embed_num-1, dim]

		text_similary = tf.reduce_sum(tf.multiply(cur_poi_text_embed, last_pois_text_embed), 2)
		reshape_text_similary = tf.reshape(text_similary, [-1, self.poi_text_embed_num - 1])

		reshape_cur_poi_text_embed = tf.reshape(cur_poi_text_embed, [-1, self.poi_text_embed_dim])
		reshape_cur_poi_text_embed = se_block(reshape_cur_poi_text_embed, self.poi_text_embed_dim,
		                                      'CurPoiTextEmbed', self.is_training, self.params['se_type'])

		return tf.concat([reshape_text_similary, reshape_cur_poi_text_embed], axis=1)


class QueryFeature(InputBase):
	def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
		super(QueryFeature, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
		self.max_query_num = 10
		self.poi_top10_query = ["list_ctr_poi_top10_query_list_int64[%d]" % i for i in range(self.max_query_num)]
		self._last_poi_query = [
			["list_last_1st_poi_query_list_int64[%d]" % i for i in range(self.max_query_num)],
			["list_last_2nd_poi_query_list_int64[%d]" % i for i in range(self.max_query_num)],
			["list_last_3rd_poi_query_list_int64[%d]" % i for i in range(self.max_query_num)],
			["list_last_1st_order_poi_query_list_int64[%d]" % i for i in range(self.max_query_num)]
		]
		self.last_query_fea_num = len(self._last_poi_query)
		self.last_poi_query = [v for l in self._last_poi_query for v in l]

		self.max_rt_query_num = 5
		self._rt_query_feature = [
			["query_latest_1hour_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_latest_1day_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_category_latest_1hour_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_category_latest_1day_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_second_category_latest_1hour_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_second_category_latest_1day_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_first_category_latest_1hour_str_int64[%d]" % i for i in range(self.max_rt_query_num)],
			["query_first_category_latest_1day_str_int64[%d]" % i for i in range(self.max_rt_query_num)]
		]
		self.rt_query_fea_num = len(self._rt_query_feature)
		self.rt_query_feature = [v for l in self._rt_query_feature for v in l]

		self.cat_list = [self.poi_top10_query, self.last_poi_query, self.rt_query_feature]

	def recieve_gather_features(self, cat_feas, dense_feas):
		# cat_feas
		self.cat_fea_split = cat_feas
		self.cat_fea_split_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

	def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
		poi_query_fea_emb, last_poi_query_fea_emb, query_rt_fea_emb = self.cat_fea_split_emb

		# poi top 10 query
		poi_query_fea_emb = tf.reshape(poi_query_fea_emb, [-1, self.max_query_num * embed_dim])
		poi_query_fea_emb_se = se_block(poi_query_fea_emb, embed_dim, 'poi_query_fea', self.is_training, self.params['se_type'])

		# last poi query
		last_poi_query_fea_emb = tf.reshape(last_poi_query_fea_emb, [-1, self.last_query_fea_num, self.max_query_num * embed_dim])

		# poi & last poid similarity
		reshape_poi_query_fea_emb = tf.reshape(poi_query_fea_emb, [-1, 1, self.max_query_num * embed_dim])
		poi_query_similarity = tf.reduce_sum(tf.multiply(reshape_poi_query_fea_emb, last_poi_query_fea_emb), axis=2)
		poi_query_similarity = tf.reshape(poi_query_similarity, [-1, self.last_query_fea_num])

		# real time query feature
		query_rt_fea_emb = tf.reshape(query_rt_fea_emb, [-1, self.max_rt_query_num, self.rt_query_fea_num, embed_dim])
		query_rt_fea_emb_pooling = tf.reduce_sum(query_rt_fea_emb, axis=1)
		reshape_query_rt_fea_emb = tf.reshape(query_rt_fea_emb_pooling, [-1, embed_dim * self.rt_query_fea_num])
		reshape_query_rt_fea_emb_se = se_block(reshape_query_rt_fea_emb, embed_dim, 'rt_query_fea', self.is_training, self.params['se_type'])

		return tf.concat([poi_query_fea_emb_se, poi_query_similarity, reshape_query_rt_fea_emb_se], axis=1)

# class PretrainPoiEmbed(InputBase):
#	def __init__(self, data_struct, params, is_training):
#		super(PretrainPoiEmbed, self).__init__(data_struct, params, is_training)
#		self.embed_fea_num = 4
#		self.pretrain_embed_dim = 32
#		self.pretrain_poi_feat_list = ["list_poi_id_pretrain_embedding_for_self_v2",
#									"list_poi_id_pretrain_embedding_for_list_last_1st_click_v2",
#									"list_poi_id_pretrain_embedding_for_list_last_2nd_click_v2",
#									"list_poi_id_pretrain_embedding_for_list_last_3rd_click_v2"]
#		self.pretrain_poi_feat_list = [
#			["list_poi_id_pretrain_embedding_for_self_v2[%d]" % i for i in range(self.pretrain_embed_dim)],
#			["list_poi_id_pretrain_embedding_for_list_last_1st_click_v2[%d]" % i for i in range(self.pretrain_embed_dim)],
#			["list_poi_id_pretrain_embedding_for_list_last_2nd_click_v2[%d]" % i for i in range(self.pretrain_embed_dim)],
#			["list_poi_id_pretrain_embedding_for_list_last_3rd_click_v2[%d]" % i for i in range(self.pretrain_embed_dim)]]
#		self.pretrain_poi_feat = [v for l in self.pretrain_poi_feat_list for v in l]
#
#	def __call__(self, cat_features_embed, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
#		feat_index = index_of_tensor(self.total_columns_info.index_of_column, self.pretrain_poi_feat)
#		pretrain_poi_embed = tf.gather(dense_features, feat_index, axis=1)
#		pretrain_poi_embed = tf.reshape(pretrain_poi_embed, [-1, self.embed_fea_num, self.pretrain_embed_dim])
#
#		cur_poi_embed = tf.slice(pretrain_poi_embed, [0, 0, 0], [-1, 1, -1])
#		last_pois_embed = tf.slice(pretrain_poi_embed, [0, 1, 0], [-1, -1, -1])
#
#		similary = tf.reduce_sum(tf.multiply(cur_poi_embed, last_pois_embed), 2)
#		reshape_similary = tf.reshape(similary, [-1, self.embed_feat_num - 1])
#
#		return reshape_similary
