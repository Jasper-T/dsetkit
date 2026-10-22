import pytest

from dsetkit.annotations.io import dump, load
from dsetkit.annotations.schema import Annotation, AnnotationItem, BBox


@pytest.mark.parametrize("target", ["voc", "labelme", "yolo"])
def test_fractional_detection_box_survives_voc_round_trip(tmp_path, target):
    coords = [10.125, 20.25, 10.375, 20.5]
    annotation = Annotation(
        width=60,
        height=40,
        names=["object"],
        items=[AnnotationItem(category="object", category_id=0, bbox=BBox(*coords))],
    )
    voc = tmp_path / "sample.xml"
    dump(annotation, str(voc), fmt="voc")
    restored = load(label_path=str(voc), fmt="voc", names=["object"])
    suffix = {"voc": ".xml", "labelme": ".json", "yolo": ".txt"}[target]
    output = tmp_path / f"converted{suffix}"
    dump(restored, str(output), fmt=target)
    final = load(label_path=str(output), fmt=target, names=["object"], width=60, height=40)
    bbox = final.items[0].bbox
    assert [bbox.x1, bbox.y1, bbox.x2, bbox.y2] == pytest.approx(coords)
    assert bbox.x2 > bbox.x1 and bbox.y2 > bbox.y1
