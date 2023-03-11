
import utils.CONST as cst
import itertools as it

print "?????"
def parse_inputs_unit(input_units):
    # input "handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense"
    # "input_units": ["handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense"],

    def one_unit(unit):
        f_name, m_names = unit.split(":")
        m_name_list = m_names.split(",")
        return [".".join([cst.handle_lib_path, f_name, m]) for m in m_name_list]
    rs = map(one_unit,input_units)
    return list(it.chain(*rs))


input_units = ["handle_cat_dense_unit:Dense,Category,PoiTextEmbedDense",
                   "handle_cat_dense_unit2:Dense,Category,PoiTextEmbedDense"]

print parse_inputs_unit(input_units)
