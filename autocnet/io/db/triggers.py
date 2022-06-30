from sqlalchemy.schema import DDL
from ... import sql

valid_geom_function = DDL(sql.valid_geom_func_str)

valid_geom_trigger = DDL(sql.valid_geom_trig_str)

valid_point_function = DDL(sql.valid_point_func_str)

valid_point_trigger = DDL(sql.valid_point_trig_str)

ignore_image_function = DDL(sql.ignore_image_fun_str)

ignore_image_trigger = DDL(sql.ignore_image_trig_str)


# several funcs and an operator needed to get json diff working. 
jsonb_delete_func = DDL(sql.json_delete_func_str)


def generate_history_triggers(table):
  tablename = table.__tablename__

  history_update_function = DDL(sql.history_update_func_str.format(formatInput=tablename))

  history_insert_function = DDL(sql.history_insert_func_str.format(formatInput=tablename))

  history_delete_function = DDL(sql.history_delete_func_str.format(formatInput=tablename))

  history_insert_trigger = DDL(sql.history_insert_trig_str.format(formatInput=tablename))

  history_delete_trigger = DDL(sql.history_delete_trig_str.format(formatInput=tablename))

  history_update_trigger = DDL(sql.history_update_trig_str.format(formatInput=tablename))

  return history_update_function, history_insert_function, history_delete_function, history_insert_trigger, history_delete_trigger, history_update_trigger 