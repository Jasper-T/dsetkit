from dsetkit import Dataset
from dsetkit.tools import convert_dataset, plot_dataset


dataset = Dataset(
    image_dir="/workspace/projects/00_datasets/phone/images",
    label_dir="/workspace/projects/00_datasets/phone/labelme",
    names=["phone"],
    source_format="labelme",
)
dataset.build()

convert_dataset(dataset, target_format="yolo", names=["phone"])
plot_dataset(dataset, names=["phone"])

