from .annotations.schema import (
    ClassSchemaComparison,
    ClassSchemaDifference,
    compare_class_schemas,
)
from .dataset import Dataset
from .collection import DatasetCollection, open_dataset
from .collection_profile import DatasetCollectionProfile
from .profile import DatasetProfile, DatasetProfileError, DatasetSample
from .detection import AnnotationTarget, EvaluationSample, PredictionResult
from .discovery import (
    AnnotationSourceError,
    DatasetDiscovery,
    DatasetDiscoveryCandidate,
    DatasetPart,
    UnsupportedDatasetLayoutError,
    discover_parts,
    select_annotation_source,
)

__all__ = [
    "AnnotationTarget",
    "ClassSchemaComparison",
    "ClassSchemaDifference",
    "Dataset",
    "DatasetDiscovery",
    "DatasetDiscoveryCandidate",
    "DatasetProfile",
    "DatasetCollectionProfile",
    "DatasetProfileError",
    "DatasetSample",
    "EvaluationSample",
    "PredictionResult",
    "compare_class_schemas",
    "DatasetCollection",
    "DatasetPart",
    "AnnotationSourceError",
    "UnsupportedDatasetLayoutError",
    "discover_parts",
    "select_annotation_source",
    "open_dataset",
    "partition_paths",
    "combine_partitions",
    "save_partition_txts",
]


from .partitions import combine_partitions, partition_paths, save_partition_txts
