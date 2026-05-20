# --------------------------------------------------------------------------------
# 标签格式转换
# 支持: 'yolo'(txt), 'voc'(xml), 'labelme'(json) 相互转换
# --------------------------------------------------------------------------------

from dsetkit.annotations.io import load, dump, convert


# 1. 推荐使用
# 特别注意：当target_format为yolo时，names参数必须指定，否则会报错
# 特别注意：当target_format为yolo时，names参数必须指定，否则会报错
# 特别注意：当target_format为yolo时，names参数必须指定，否则会报错

convert(
    label_path="your_label.json",
    source_format="labelme",
    target_format="yolo",
    names=["name"],
    out_dir="your_output_dir"
)


# 2. debug使用,查看中间转换
# 特别注意：当load的fmt为yolo时，names参数必须指定，否则会报错
# 特别注意：当load的fmt为yolo时，names参数必须指定，否则会报错
# 特别注意：当load的fmt为yolo时，names参数必须指定，否则会报错
ann = load(
    label_path="your_label.json",
    image_path="your_image.jpg",
    fmt="labelme",
    names=["name"]
)

dump(ann, "examples/1773936371456.txt", names=[],fmt="yolo")


