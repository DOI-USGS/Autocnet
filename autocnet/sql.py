# This file is to centralize all the sql strings that are used throughout autocnet
# into one place

compute_overlaps_sql = """
WITH intersectiongeom AS
(SELECT geom AS geom FROM ST_Dump((
   SELECT ST_Polygonize(the_geom) AS the_geom FROM (
     SELECT ST_Union(the_geom) AS the_geom FROM (
     SELECT ST_ExteriorRing((ST_DUMP(geom)).geom) AS the_geom
       FROM images WHERE images.geom IS NOT NULL) AS lines
  ) AS noded_lines))),
iid AS (
 SELECT images.id, intersectiongeom.geom AS geom
    FROM images, intersectiongeom
    WHERE images.geom is NOT NULL AND
    ST_INTERSECTS(intersectiongeom.geom, images.geom) AND
    ST_AREA(ST_INTERSECTION(intersectiongeom.geom, images.geom)) > 0.000001
)
INSERT INTO overlay(intersections, geom) SELECT row.intersections, row.geom FROM
(SELECT iid.geom, array_agg(iid.id) AS intersections
  FROM iid GROUP BY iid.geom) AS row WHERE array_length(intersections, 1) > 1;
"""

select_ten_pub_image = 'SELECT * FROM public.images LIMIT 10'

select_pub_image = 'SELECT * FROM public.images'

from_database_composite = '''WITH i as ({formatInput}) SELECT i1.id
        as i1_id,i1.path as i1_path, i2.id as i2_id, i2.path as i2_path
        FROM i  as i1, i as i2
        WHERE ST_INTERSECTS(i1.geom, i2.geom) = TRUE
        AND i1.id < i2.id'''

db_to_df_sql_string = """
SELECT measures."pointid",
        points."pointType",
        points."apriori",
        points."adjusted",
        points."pointIgnore",
        points."referenceIndex",
        points."identifier",
        measures."id",
        measures."serialnumber",
        measures."sample",
        measures."line",
        measures."measureType",
        measures."imageid",
        measures."measureIgnore",
        measures."measureJigsawRejected",
        measures."aprioriline",
        measures."apriorisample"
FROM measures
INNER JOIN points ON measures."pointid" = points."id"
WHERE
    points."pointIgnore" = False AND
    measures."measureIgnore" = FALSE AND
    measures."measureJigsawRejected" = FALSE AND
    measures."imageid" NOT IN
        (SELECT measures."imageid"
        FROM measures
        INNER JOIN points ON measures."pointid" = points."id"
        WHERE measures."measureIgnore" = False and measures."measureJigsawRejected" = False AND points."pointIgnore" = False
        GROUP BY measures."imageid"
        HAVING COUNT(DISTINCT measures."pointid")  < 3)
ORDER BY measures."pointid", measures."id";
"""


valid_geom_func_str = """
CREATE OR REPLACE FUNCTION validate_geom()
  RETURNS trigger AS
$BODY$
  BEGIN
      NEW.geom = ST_MAKEVALID(NEW.geom);
      RETURN NEW;
    EXCEPTION WHEN OTHERS THEN
      NEW.ignore = true;
      RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
"""

valid_geom_trig_str = """
CREATE TRIGGER image_inserted
  BEFORE INSERT OR UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE validate_geom();
"""

valid_point_func_str ="""
CREATE OR REPLACE FUNCTION validate_points()
  RETURNS trigger AS
$BODY$
BEGIN
 IF (SELECT COUNT(*)
	 FROM MEASURES
	 WHERE pointid = NEW.pointid AND "measureIgnore" = False) < 2
 THEN
   UPDATE points
     SET "pointIgnore" = True
	 WHERE points.id = NEW.pointid;
 ELSE
   UPDATE points
   SET "pointIgnore" = False
   WHERE points.id = NEW.pointid;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
"""

valid_point_trig_str = """
CREATE TRIGGER active_measure_changes
  AFTER UPDATE
  ON measures
  FOR EACH ROW
EXECUTE PROCEDURE validate_points();
"""

ignore_image_fun_str = """
CREATE OR REPLACE FUNCTION ignore_image()
  RETURNS trigger AS
$BODY$
BEGIN
 IF NEW.ignore
 THEN
   UPDATE measures
     SET "measureIgnore" = True
     WHERE measures.serialnumber = NEW.serial;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
"""


ignore_image_trig_str = """
CREATE TRIGGER image_ignored
  AFTER UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE ignore_image();
"""

json_delete_func_str = """
SET search_path = 'public';

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b text) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE key <> b
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, text) IS 'delete key in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = text);
COMMENT ON OPERATOR - (jsonb, text) IS 'delete key from left operand';

--

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b text[]) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE key <> ALL(b)
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, text[]) IS 'delete keys in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = text[]);
COMMENT ON OPERATOR - (jsonb, text[]) IS 'delete keys from left operand';

--

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b jsonb) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE NOT ('{' || to_json(key) || ':' || value || '}')::jsonb <@ b
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, jsonb) IS 'delete matching pairs in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = jsonb);
COMMENT ON OPERATOR - (jsonb, jsonb) IS 'delete matching pairs from left operand';
"""

history_update_func_str = """
  CREATE OR REPLACE FUNCTION {formatInput}_history_update()
  RETURNS TRIGGER AS $$
  DECLARE
    js_new jsonb := row_to_json(NEW)::jsonb;
    js_old jsonb := row_to_json(OLD)::jsonb;
  BEGIN
    INSERT INTO {formatInput}_history(fk, "eventTime", "executedBy", event, before, after)
      VALUES((js_old->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'update', js_old - js_new, js_new - js_old);
    RETURN NEW;
  END;

  $$ LANGUAGE plpgsql;
  """

history_insert_func_str = """
  CREATE OR REPLACE FUNCTION {formatInput}_history_insert()
  RETURNS TRIGGER AS $$
  DECLARE 
    js_new jsonb := row_to_json(NEW)::jsonb;
  BEGIN
    INSERT INTO {formatInput}_history(fk, "eventTime", "executedBy", event, after)
       VALUES((js_new->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'insert', js_new);
    RETURN NEW;
  END;
  $$ LANGUAGE plpgsql;
  """

history_delete_func_str = """
  CREATE OR REPLACE FUNCTION {formatInput}_history_delete()
  RETURNS TRIGGER AS $$
  DECLARE
    js_old jsonb := row_to_json(OLD)::jsonb;
  BEGIN
    INSERT INTO {formatInput}_history(fk, "eventTime", "executedBy", event, before)
       VALUES((js_old->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'delete', js_old);
    RETURN NEW;
  END;
  $$ LANGUAGE plpgsql;
  """

history_insert_trig_str = """
  CREATE TRIGGER {formatInput}_history_insert AFTER INSERT ON {formatInput}
    FOR EACH ROW EXECUTE PROCEDURE {formatInput}_history_insert();
  """

history_delete_trig_str = """
  CREATE TRIGGER {formatInput}_history_delete AFTER DELETE ON {formatInput}
    FOR EACH ROW EXECUTE PROCEDURE {formatInput}_history_delete();
  """

history_update_trig_str = """
  CREATE TRIGGER {formatInput}_history_update AFTER UPDATE ON {formatInput}
    FOR EACH ROW EXECUTE PROCEDURE {formatInput}_history_update();
  """