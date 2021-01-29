import paddlehub as hub
import paddlex as pdx
from paddlex.det import transforms

def det():
    
    pass
def to_voc(fname, obj_type=[], pos=[], res=[1920,1080]):
    if obj_type != []:
        objs = [
        """
        <object>
            <name>{}</name>
            <bndbox>
              <xmin>{}</xmin>
              <ymin>{}</ymin>
              <xmax>{}</xmax>
              <ymax>{}</ymax>
            </bndbox>
        </object>
        """.format(t, p[0], p[1], p[2], p[3]) for t,p in zip(obj_type, pos)
        ]
        objs = "".join(objs)
    else:
        objs = ""

    res = """<?xml version='1.0' encoding='UTF-8'?>
    <annotation>
      <filename>{}</filename>
      <object_num>{}</object_num>
      <size>
        <width>{}</width>
        <height>{}</height>
      </size>
        {}
    </annotation>""".format(fname, len(obj_type), res[0], res[1], objs)


    return res
