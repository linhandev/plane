def to_voc(name, xmin,ymin,xmax,ymax):
    res = """<?xml version='1.0' encoding='UTF-8'?>
    <annotation>
      <filename>{}</filename>
      <object_num>1</object_num>
      <size>
        <width>1920</width>
        <height>1080</height>
      </size>
      <object>
        <name>前起落架</name>
        <bndbox>
          <xmin>{}</xmin>
          <ymin>{}</ymin>
          <xmax>{}</xmax>
          <ymax>{}</ymax>
        </bndbox>
      </object>
    </annotation>""".format("1.png", 1,2,3,4)
    return res
