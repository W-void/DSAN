from utils.tools import merge

a = {
	"CateSearchByName": "uuid_period_clk_poi_position_list_int64,poi_top_sale_spu_name_int64,list_ctr_sess_index_list_10_v3_int64,ec_post_nt_poi_id_list,uuid_click_wm_poi_list_12hour_idr_int64,cat_keys,uuid_category_clk_poi_timestamp_list,pathway_tgt_acttype_list,list_last_1st_poi_query_list_int64,uuid_clk_poi_timestamp_list_int64,query_latest_1day_str_int64,pathway_tgt_poi_list#map_attention",
}
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
rs = [merge(a, x) for x in rs]
print rs
