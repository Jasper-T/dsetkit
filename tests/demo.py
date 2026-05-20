import cv2
from pathlib import Path

from dsetkit.annotations.io import load, dump, convert
from dsetkit.visualize.plot import Plotter


input_dir = Path("examples")
output_dir = Path("examples")

im_path = input_dir / "1773936371456.jpg"
json_path = input_dir / "1773936371456.json"
xml_path = input_dir / "1773936371456.xml"
txt_path = input_dir / "1773936371456.txt"

img = cv2.imread(im_path)

# labelme -> voc
fmt = "labelme"
target_format = "voc"
convert(
    label_path=json_path,
    image_path=im_path,
    target_format=target_format,
    fmt=fmt,
    out_dir=output_dir,
)   

# labelme -> yolo
fmt = "labelme"
target_format = "yolo"
convert(
    label_path=json_path,
    image_path=im_path,
    target_format=target_format,
    fmt=fmt,
    out_dir=output_dir,
    names=["smoke"]
)


# labelme schema
fmt = "labelme"
plotter = Plotter(img)

ann = load(
    label_path=json_path,
    image_path=im_path,
    fmt="labelme"
)

for item in ann.items:
    plotter.detection_from_schema(item)

plotter.save(output_dir / f"{fmt}.jpg")

# voc schema
fmt = "voc"
plotter = Plotter(img)

ann = load(
    label_path=xml_path,
    image_path=im_path,
    fmt="voc"
)

for item in ann.items:
    plotter.detection_from_schema(item)

plotter.save(output_dir / f"{fmt}.jpg")

# txt schema

fmt = "yolo"
plotter = Plotter(img)

ann = load(
    label_path=txt_path,
    image_path=im_path,
    fmt="yolo",
    names=["smoke"]
)

for item in ann.items:
    plotter.detection_from_schema(item)

plotter.save(output_dir / f"{fmt}.jpg")

