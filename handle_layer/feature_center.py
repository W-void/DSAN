# from handle_layer.handle_lib.handle_cat_dense_unit import Dense,PoiTextEmbedDense,Category
import json
from collections import Counter

dense_feat_list = ["distance", "user_tag_pref_click_3day", "user_tag_pref_click_15day", "user_tag_pref_click_30day", "user_tag_pref_order_30day", "uv_cvr_15day", "pv_ctr_7day",
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
cate_feat_list = ["discrete_slot_int64", "discrete_poi_position_int64", "discrete_client_type_int64", "discrete_app_version_int64", "discrete_hour_of_day_int64",
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
poi_text_feat = ["list_poi_id_text_embedding_for_self_v2[0]", "list_poi_id_text_embedding_for_self_v2[1]", "list_poi_id_text_embedding_for_self_v2[2]",
                 "list_poi_id_text_embedding_for_self_v2[3]", "list_poi_id_text_embedding_for_self_v2[4]", "list_poi_id_text_embedding_for_self_v2[5]",
                 "list_poi_id_text_embedding_for_self_v2[6]", "list_poi_id_text_embedding_for_self_v2[7]", "list_poi_id_text_embedding_for_self_v2[8]",
                 "list_poi_id_text_embedding_for_self_v2[9]", "list_poi_id_text_embedding_for_self_v2[10]", "list_poi_id_text_embedding_for_self_v2[11]",
                 "list_poi_id_text_embedding_for_self_v2[12]", "list_poi_id_text_embedding_for_self_v2[13]", "list_poi_id_text_embedding_for_self_v2[14]",
                 "list_poi_id_text_embedding_for_self_v2[15]", "list_poi_id_text_embedding_for_self_v2[16]", "list_poi_id_text_embedding_for_self_v2[17]",
                 "list_poi_id_text_embedding_for_self_v2[18]", "list_poi_id_text_embedding_for_self_v2[19]", "list_poi_id_text_embedding_for_self_v2[20]",
                 "list_poi_id_text_embedding_for_self_v2[21]", "list_poi_id_text_embedding_for_self_v2[22]", "list_poi_id_text_embedding_for_self_v2[23]",
                 "list_poi_id_text_embedding_for_self_v2[24]", "list_poi_id_text_embedding_for_self_v2[25]", "list_poi_id_text_embedding_for_self_v2[26]",
                 "list_poi_id_text_embedding_for_self_v2[27]", "list_poi_id_text_embedding_for_self_v2[28]", "list_poi_id_text_embedding_for_self_v2[29]",
                 "list_poi_id_text_embedding_for_self_v2[30]", "list_poi_id_text_embedding_for_self_v2[31]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[0]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[1]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[2]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[3]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[4]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[5]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[6]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[7]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[8]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[9]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[10]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[11]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[12]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[13]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[14]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[15]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[16]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[17]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[18]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[19]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[20]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[21]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[22]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[23]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[24]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[25]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[26]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[27]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[28]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[29]", "list_poi_id_text_embedding_for_list_last_1st_click_v2[30]",
                 "list_poi_id_text_embedding_for_list_last_1st_click_v2[31]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[0]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[1]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[2]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[3]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[4]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[5]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[6]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[7]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[8]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[9]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[10]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[11]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[12]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[13]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[14]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[15]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[16]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[17]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[18]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[19]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[20]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[21]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[22]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[23]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[24]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[25]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[26]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[27]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[28]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[29]", "list_poi_id_text_embedding_for_list_last_2nd_click_v2[30]",
                 "list_poi_id_text_embedding_for_list_last_2nd_click_v2[31]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[0]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[1]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[2]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[3]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[4]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[5]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[6]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[7]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[8]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[9]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[10]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[11]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[12]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[13]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[14]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[15]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[16]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[17]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[18]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[19]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[20]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[21]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[22]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[23]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[24]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[25]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[26]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[27]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[28]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[29]", "list_poi_id_text_embedding_for_list_last_3rd_click_v2[30]",
                 "list_poi_id_text_embedding_for_list_last_3rd_click_v2[31]"]
rerank_feat_list = ["rerank_top40_ctr_list[0]", "rerank_top40_ctr_list[1]", "rerank_top40_ctr_list[2]", "rerank_top40_ctr_list[3]", "rerank_top40_ctr_list[4]", "rerank_top40_ctr_list[5]",
                    "rerank_top40_ctr_list[6]", "rerank_top40_ctr_list[7]", "rerank_top40_ctr_list[8]", "rerank_top40_ctr_list[9]", "rerank_top40_ctr_list[10]", "rerank_top40_ctr_list[11]",
                    "rerank_top40_ctr_list[12]", "rerank_top40_ctr_list[13]", "rerank_top40_ctr_list[14]", "rerank_top40_ctr_list[15]", "rerank_top40_ctr_list[16]", "rerank_top40_ctr_list[17]",
                    "rerank_top40_ctr_list[18]", "rerank_top40_ctr_list[19]", "rerank_top40_ctr_list[20]", "rerank_top40_ctr_list[21]", "rerank_top40_ctr_list[22]", "rerank_top40_ctr_list[23]",
                    "rerank_top40_ctr_list[24]", "rerank_top40_ctr_list[25]", "rerank_top40_ctr_list[26]", "rerank_top40_ctr_list[27]", "rerank_top40_ctr_list[28]", "rerank_top40_ctr_list[29]",
                    "rerank_top40_ctr_list[30]", "rerank_top40_ctr_list[31]", "rerank_top40_ctr_list[32]", "rerank_top40_ctr_list[33]", "rerank_top40_ctr_list[34]", "rerank_top40_ctr_list[35]",
                    "rerank_top40_ctr_list[36]", "rerank_top40_ctr_list[37]", "rerank_top40_ctr_list[38]", "rerank_top40_ctr_list[39]", "rerank_top40_cvr_list[0]", "rerank_top40_cvr_list[1]",
                    "rerank_top40_cvr_list[2]", "rerank_top40_cvr_list[3]", "rerank_top40_cvr_list[4]", "rerank_top40_cvr_list[5]", "rerank_top40_cvr_list[6]", "rerank_top40_cvr_list[7]",
                    "rerank_top40_cvr_list[8]", "rerank_top40_cvr_list[9]", "rerank_top40_cvr_list[10]", "rerank_top40_cvr_list[11]", "rerank_top40_cvr_list[12]", "rerank_top40_cvr_list[13]",
                    "rerank_top40_cvr_list[14]", "rerank_top40_cvr_list[15]", "rerank_top40_cvr_list[16]", "rerank_top40_cvr_list[17]", "rerank_top40_cvr_list[18]", "rerank_top40_cvr_list[19]",
                    "rerank_top40_cvr_list[20]", "rerank_top40_cvr_list[21]", "rerank_top40_cvr_list[22]", "rerank_top40_cvr_list[23]", "rerank_top40_cvr_list[24]", "rerank_top40_cvr_list[25]",
                    "rerank_top40_cvr_list[26]", "rerank_top40_cvr_list[27]", "rerank_top40_cvr_list[28]", "rerank_top40_cvr_list[29]", "rerank_top40_cvr_list[30]", "rerank_top40_cvr_list[31]",
                    "rerank_top40_cvr_list[32]", "rerank_top40_cvr_list[33]", "rerank_top40_cvr_list[34]", "rerank_top40_cvr_list[35]", "rerank_top40_cvr_list[36]", "rerank_top40_cvr_list[37]",
                    "rerank_top40_cvr_list[38]", "rerank_top40_cvr_list[39]", "rerank_top40_cxr_list[0]", "rerank_top40_cxr_list[1]", "rerank_top40_cxr_list[2]", "rerank_top40_cxr_list[3]",
                    "rerank_top40_cxr_list[4]", "rerank_top40_cxr_list[5]", "rerank_top40_cxr_list[6]", "rerank_top40_cxr_list[7]", "rerank_top40_cxr_list[8]", "rerank_top40_cxr_list[9]",
                    "rerank_top40_cxr_list[10]", "rerank_top40_cxr_list[11]", "rerank_top40_cxr_list[12]", "rerank_top40_cxr_list[13]", "rerank_top40_cxr_list[14]", "rerank_top40_cxr_list[15]",
                    "rerank_top40_cxr_list[16]", "rerank_top40_cxr_list[17]", "rerank_top40_cxr_list[18]", "rerank_top40_cxr_list[19]", "rerank_top40_cxr_list[20]", "rerank_top40_cxr_list[21]",
                    "rerank_top40_cxr_list[22]", "rerank_top40_cxr_list[23]", "rerank_top40_cxr_list[24]", "rerank_top40_cxr_list[25]", "rerank_top40_cxr_list[26]", "rerank_top40_cxr_list[27]",
                    "rerank_top40_cxr_list[28]", "rerank_top40_cxr_list[29]", "rerank_top40_cxr_list[30]", "rerank_top40_cxr_list[31]", "rerank_top40_cxr_list[32]", "rerank_top40_cxr_list[33]",
                    "rerank_top40_cxr_list[34]", "rerank_top40_cxr_list[35]", "rerank_top40_cxr_list[36]", "rerank_top40_cxr_list[37]", "rerank_top40_cxr_list[38]", "rerank_top40_cxr_list[39]",
                    "rerank_top40_vs_list[0]", "rerank_top40_vs_list[1]", "rerank_top40_vs_list[2]", "rerank_top40_vs_list[3]", "rerank_top40_vs_list[4]", "rerank_top40_vs_list[5]",
                    "rerank_top40_vs_list[6]", "rerank_top40_vs_list[7]", "rerank_top40_vs_list[8]", "rerank_top40_vs_list[9]", "rerank_top40_vs_list[10]", "rerank_top40_vs_list[11]",
                    "rerank_top40_vs_list[12]", "rerank_top40_vs_list[13]", "rerank_top40_vs_list[14]", "rerank_top40_vs_list[15]", "rerank_top40_vs_list[16]", "rerank_top40_vs_list[17]",
                    "rerank_top40_vs_list[18]", "rerank_top40_vs_list[19]", "rerank_top40_vs_list[20]", "rerank_top40_vs_list[21]", "rerank_top40_vs_list[22]", "rerank_top40_vs_list[23]",
                    "rerank_top40_vs_list[24]", "rerank_top40_vs_list[25]", "rerank_top40_vs_list[26]", "rerank_top40_vs_list[27]", "rerank_top40_vs_list[28]", "rerank_top40_vs_list[29]",
                    "rerank_top40_vs_list[30]", "rerank_top40_vs_list[31]", "rerank_top40_vs_list[32]", "rerank_top40_vs_list[33]", "rerank_top40_vs_list[34]", "rerank_top40_vs_list[35]",
                    "rerank_top40_vs_list[36]", "rerank_top40_vs_list[37]", "rerank_top40_vs_list[38]", "rerank_top40_vs_list[39]", "rerank_top40_cs_list[0]", "rerank_top40_cs_list[1]",
                    "rerank_top40_cs_list[2]", "rerank_top40_cs_list[3]", "rerank_top40_cs_list[4]", "rerank_top40_cs_list[5]", "rerank_top40_cs_list[6]", "rerank_top40_cs_list[7]",
                    "rerank_top40_cs_list[8]", "rerank_top40_cs_list[9]", "rerank_top40_cs_list[10]", "rerank_top40_cs_list[11]", "rerank_top40_cs_list[12]", "rerank_top40_cs_list[13]",
                    "rerank_top40_cs_list[14]", "rerank_top40_cs_list[15]", "rerank_top40_cs_list[16]", "rerank_top40_cs_list[17]", "rerank_top40_cs_list[18]", "rerank_top40_cs_list[19]",
                    "rerank_top40_cs_list[20]", "rerank_top40_cs_list[21]", "rerank_top40_cs_list[22]", "rerank_top40_cs_list[23]", "rerank_top40_cs_list[24]", "rerank_top40_cs_list[25]",
                    "rerank_top40_cs_list[26]", "rerank_top40_cs_list[27]", "rerank_top40_cs_list[28]", "rerank_top40_cs_list[29]", "rerank_top40_cs_list[30]", "rerank_top40_cs_list[31]",
                    "rerank_top40_cs_list[32]", "rerank_top40_cs_list[33]", "rerank_top40_cs_list[34]", "rerank_top40_cs_list[35]", "rerank_top40_cs_list[36]", "rerank_top40_cs_list[37]",
                    "rerank_top40_cs_list[38]", "rerank_top40_cs_list[39]"]


def get_all_seqs():
    not_used_cat = sorted(list(set(rs['features']['cat_feature']['column']) - set(cate_feat_list)))
    # print not_used_cat
    fs = Counter(map(lambda x: x.split('[')[0], not_used_cat))
    list_keys = [k for k, v in fs.items() if v > 1]
    cat_keys = [k for k, v in fs.items() if v == 1]
    return list_keys, cat_keys



# print len(list_keys), len(cat_keys)
# print list_keys
# print cat_keys
# print sorted(not_used_cat,key=lambda x:x.split(''))

# print sorted(filter(lambda x:"rerank" not in x,list(set(rs['features']['dense_feature']['column']) - set(dense_feat_list) -set(poi_text_feat))))


# print filter(lambda x:"rerank" in x,list(set(rs['features']['dense_feature']['column']) ))

# pooling/f_pooling/din-query poi/din-query poi-tag
if __name__ == '__main__':
    rs = json.load(open('/Users/dongjian/2022/waimai_ad_offline_rec_predict/exp/user/base/data_struct.json'))
    total_cat_feautre = rs['features']['cat_feature']['column']

    print rs.keys(), len(rs['totalFeatures']), len(rs['features']), rs['features'].keys()
    print "total cat feature length is {}, dense feature length is {}".format(len(rs['features']['cat_feature']['column']), len(rs['features']['dense_feature']['column']))
    print "already use cat feature {}, dense feature {} poi_text_feat {} rerank_feat_list {}".format(len(cate_feat_list), len(dense_feat_list), len(poi_text_feat), len(rerank_feat_list))

    list_keys, cat_keys = get_all_seqs()
    print list_keys
    cand_keys = {x: [r for r in rs['features']['cat_feature']['column'] if x in r] for x in list_keys}
    print cand_keys, len(cat_keys)
