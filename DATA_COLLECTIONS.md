# Dataset discovery, profiles and partitions

The SDK separates source scanning, immutable analysis results, processing and path-based partitions. API changes are listed below.

## Recommended entry points

```python
from dsetkit import (
    discover_parts, select_annotation_source, open_dataset, save_partition_txts,
)

native, parts = discover_parts("dataset")
for part in parts:
    source = select_annotation_source(part.discovery, requested="yolo")
    print(part.split, part.discovery.image_dir, source)

collection = open_dataset("dataset", task_type="detection", names=["object"], source_format="yolo")
repartitioned = collection.split((0.8, 0.2, 0), seed=22)
profile = repartitioned.profile()
print(profile.as_dict())
# Explicit TXT output is separate from logical collection operations.
save_partition_txts({
    split: [sample.image_path for sample in profile if profile.members[sample] == split]
    for split in ("train", "val", "test")
}, "output")
```

## Fixed-depth discovery

`discover_parts(root)` recognizes the following fixed scan locations. It never
recursively searches for a dataset or scans nested image/annotation directories.

| Layout | Images | Annotations |
| --- | --- | --- |
| Flat | `root/images` | `root/<annotation-dir>` |
| Split first | `root/{train,val,test}/images` | `root/{train,val,test}/<annotation-dir>` |
| Split last | `root/images/{train,val,test}` | `root/<annotation-dir>/{train,val,test}` |

A single internal discovery implementation handles annotation-directory aliases,
case-insensitive filesystem aliases and suffix filtering for all layouts. Every
source is represented in `part.discovery.candidates`, including a unique source.
There is no second selected-source representation on discovery results.
`select_annotation_source()` selects a unique explicit format, auto-selects a
unique source when format is omitted, and rejects ambiguity. A missing explicit
format match is an error in `select_annotation_source()`. `open_dataset()` keeps a
part without any annotation candidates as image-only, even when a format is requested.

Split-first layouts take precedence if both native layouts exist. Additional
flat images beside split-first parts remain unassigned. Split roots without
`images/`, mixed direct split-root files, unrelated directories, nested files,
sidecars and directories with annotation suffixes are excluded. No supported
layout raises `UnsupportedDatasetLayoutError`. Filesystem errors, including
missing mounts, retain their original exception types.

## Sources, collections and result lifetimes

`Dataset` scans one explicit image directory, an optional label directory and a
concrete label format. It never discovers layouts or assigns train/val/test.
`Dataset(...).build().profile()` returns an immutable `DatasetProfile` containing
source facts, raw/valid/error samples, parsed annotations and optional checked image sizes.

`DatasetCollection.open(...)` orchestrates fixed-depth discovery and temporary
Dataset scans. `open_dataset(...)` delegates to it and always returns a collection,
including in explicit-directory mode. Root and explicit directories are mutually
exclusive; an explicit label directory requires a concrete format. Omitting the
label directory retains image-only/background sources without searching nearby paths.

`DatasetCollection([source_profile, ...], splits=["train", "val"])` organizes completed
source results. Its `profile()` is a separate `DatasetCollectionProfile`, with
`sources`, sample-level `members`, cumulative `excluded`, overall statistics and
`partition_stats` keyed by train/val/test or None for unassigned members. Empty native
partitions retain zero-valued statistics. Source profiles retain their own statistics;
partition statistics count only their effective members, while raw totals, errors
and orphan-label diagnostics belong to the overall result.

The collection never keeps Dataset instances, copies annotation caches, reads source
files or rechecks file validity. Datasets can be released as soon as their source
Profiles exist. An existing collection does not change when its former Dataset is
checked later; use the resulting source Profile to start a separate analysis.

### Permanent exclusions and repartitioning

There are two sample-error categories: source file/annotation validity errors and
duplicate image stems within an organization scope. Every error is terminal within
the collection lineage. Source Profiles supply the initial error pool. Initial
organization checks valid members only: same-directory and same-partition collisions
exclude all involved images. Multiple unassigned sources share one unassigned scope.
Different native partitions can retain matching stems.

Within a source scan, path stem conflicts are checked before annotation parsing or
image inspection. Conflicting samples retain their raw paths and permanent errors,
but their labels and images are not read or checked, including by later Dataset checks.
Collection organization checks only candidates that survived their source scan.

`new_collection = collection.split((0.8, 0.2, 0), seed=22)` pools current effective
members, excludes **all** duplicate-stem candidates before allocation, then assigns
the survivors. It returns a new snapshot sharing the same source Profiles; it never
moves files, discovers new files, reparses annotations or reads images. Class mappings
stay unchanged. The old result remains unchanged, including if allocation fails.
An empty training partition raises ValueError, including when conflicts empty the pool.

```
new exclusions = historical exclusions union newly conflicting samples
new effective members are a subset of current effective members
```

Changing ratios, seeds, or restoring saved results cannot revive exclusions.
Internal sample references use source ownership and the original image path, not
output paths or partition-qualified evaluation IDs. Physical sources are assumed
unchanged. An explicit new Dataset build scans a new source result; it does not
silently update an existing collection or its historical exclusions.

### Processing and image reads

Conversion, plotting and TXT export accept either Profile type. Iteration and
processing use only effective members. `source_profile(sample)` gives the source
Profile; `annotation(sample, image_info=None)` returns an independent pixel-coordinate
annotation using cached dimensions or explicitly supplied image metadata. If YOLO
needs pixel conversion and neither is available, it reports missing image metadata.

Image sizes are optional, never a prerequisite for creating a collection. Default
scanning only checks annotations and does not open images. Dataset.check_images()
performs container checks and full decoding at a caller-selected time, caches the
resulting sizes, replaces the Dataset's current Profile and keeps prior results
unchanged. Dataset.check_annotations() audits its cached annotations without rereading
label files. Profile and Collection have no check methods.

Conversion and plotting already have image paths. When they need image dimensions
not present in the image-check cache, they read the image header directly through
image utilities at that moment, even when annotation size metadata exists; plotting
also reads pixels for rendering. Neither needs a live Dataset, performs an automatic
full image check, or mutates a Profile. Cached annotations and member exclusions
remain unchanged. Image reads serve checking, dimension access or explicit rendering;
collection organization and evaluation do not read images.

Explicit image_info takes precedence over checked image dimensions, which take
precedence over annotation size in returned pixel annotations. Available image
dimensions also determine annotation-check bounds; missing or mismatched annotation
size alone does not invalidate a sample. Without image dimensions, checks use existing
annotation size and allow missing size. Source annotation caches and existing errors
remain unchanged.

Single-file load/convert with image_path also prefer image dimensions over annotation
size. Explicit width/height supply already known image dimensions without another read.

`output_path(sample)` uses train/val/test when a partition has one participating
source. A partition with several sources uses `dataset_1/train/...`, etc.; unassigned
sources use `dataset_1/...`, while one unassigned source uses bare filenames.
Repartitioning can change these output paths and their suffix-free evaluation IDs.
Conversion writes labels only; collection processing requires an explicit output directory.

### Persistence

```python
from dsetkit import DatasetCollection, DatasetCollectionProfile, open_dataset
from dsetkit.tools import convert_dataset

collection = open_dataset(
    task_type="detection", image_dir="dataset/my_images", label_dir="dataset/my_labels",
    names=["object"], source_format="yolo",
)
restored = DatasetCollectionProfile.from_json(collection.profile().export())
resumed = DatasetCollection(restored).split((0.8, 0.2, 0), seed=22)
convert_dataset(resumed.profile(), "voc", out_dir="output")
```

Both result types provide export()/from_json(); the reader must match the result
type. JSON schema version 2 has no legacy-schema compatibility layer. Source Profiles
save raw inventories, cached annotations, optional image-check sizes and source errors
once. Collection JSON saves source Profiles, partition membership and collection-added
exclusion records; their union with source errors restores the entire cumulative pool.
Missing/unknown references, overlapping effective/excluded samples and incomplete
coverage are rejected. Statistics are recomputed rather than stored as a second truth.

Restoration is offline and never repeats file audits. Source images are not embedded.
A restored result supports statistics, splits and val() offline; conversion or plotting
still needs image access when dimensions or pixels are unavailable. as_dict() is only
a statistics summary, including per-partition summaries for collections.

## Component responsibilities and independent use

Components are independently callable through explicit input/result contracts.
Independence does not mean absence of code dependencies: Dataset composes annotation
I/O and checks, Collection composes source Profiles, and Evaluator composes metrics.
A caller does not need to create a Dataset before loading one annotation, validating
one image, drawing boxes or evaluating existing predictions. Profiles are reusable
result contracts, not a mandatory entry point for all functions.

### Core classes and lifetime

| Class | Responsibility | Boundary and standalone scenario |
| --- | --- | --- |
| `Dataset` | Explicit-source scanning, annotation checks, optional image checks and DatasetProfile generation | No discovery, partitioning, batching or training; release it after obtaining a Profile, or retain it to check images later |
| `DatasetProfile` | Single-source inventories, cached annotations, check results and statistics | No file I/O, file checks, partition semantics or val(); reused after scanner release |
| `DatasetCollection` | Organize source Profiles, preserve native members, accumulate exclusions and split into a new result; open() orchestrates temporary scans | Profile input, no retained scanners; aggregation/split never reads or checks source files |
| `DatasetCollectionProfile` | Source references, sample membership, cumulative errors, global classes and overall/partition statistics | No duplicated annotation cache, scanners, file I/O or split operation |
| `Evaluator` | Pair explicit prediction/GT mappings and delegate numerical metrics | No Profile, path, image, file or model dependency; evaluate in-memory results directly |
| `Plotter` | Draw boxes/text onto a copy of a supplied image array; get, save or show the result | No scanner or Profile needed; no discovery, checks, partitioning or evaluation |

Dataset's core is scanning + annotation checks + image checks + Profile generation.
Dataset.val()/DatasetCollection.val() are thin GT adapters to Evaluator, not separate
metric implementations or training loaders. Batch conversion and plotting consume
results after Dataset disposal; they directly obtain dimensions/pixels as needed.

### Public data, result and error classes

| Classes | Data contract and typical use | Boundary |
| --- | --- | --- |
| `BBox`, `AnnotationItem`, `Annotation` | A box, one annotated object, and image annotations; manual construction, single-file I/O and validation | Mutable working records; annotations returned by Profiles are copies. Reserved non-detection fields do not imply supported tasks |
| `DatasetSample` | Image path and optional label path | Metadata, not decoded pixels or tensors |
| `DatasetProfileError` | Diagnostic path, kind and message | Collection.excluded associates diagnostics with permanently excluded sample identities |
| `DatasetDiscoveryCandidate`, `DatasetDiscovery`, `DatasetPart` | Candidate label source, image/source discovery result, and native partition association | Layout facts only; no content validity guarantee |
| `FormatSpec` | Format name, suffix and directory metadata | Registration metadata, not a parser or scan result |
| `ClassSchemaDifference`, `ClassSchemaComparison` | Differences and the result of strict class-schema comparison | No silent remapping or schema changes |
| `AnnotationTarget`, `PredictionResult` | GT boxes/classes and prediction boxes/classes/confidence, with empty values supported | Numeric inputs; no file/model loading. Array/ID checks are not full annotation-file validation |
| `EvaluationSample` | Opaque nonblank sample_id paired with predictions and GT | No filesystem semantics; uniqueness across samples is the caller's responsibility |
| `ClassMetrics`, `DetectionMetrics` | Per-class and overall numeric results, with dictionary conversion | No source access or printing |
| `ImageInfo`, `ImageValidationResult` | Image metadata and a single-image check result | No pixel cache or automatic change to an existing Profile |
| `InventoryDifference` | Added/removed/changed hash keys | No monitoring, rescan or automatic collection update |
| `AnnotationParseError`, `UnsupportedDatasetLayoutError`, `AnnotationSourceError`, `PartitionError` | Distinguish parser, layout, source-selection and partition failures | Exception contracts, not processing stages |

### Functional components

| Entry points | Independent scenario | Boundary and side effects |
| --- | --- | --- |
| `annotations.io.load/dump/convert` and label-path helpers | Read, write or convert one specified annotation | No Dataset/Profile required; output helpers may create directories, conversion writes target labels and reads dimensions when needed |
| `annotations.io.get_label_dir_name/get_label_file_suffix/new_label_path/new_label_dir/auto_label_path` | Query format paths or prepare label output paths | new_label_dir/auto_label_path create directories; no sample discovery |
| `annotations.formats.load_labelme/dump_labelme/load_voc/dump_voc/load_yolo/dump_yolo` | Direct format-specific annotation I/O | Usually called through uniform load/dump; dimensions and class mapping remain explicit inputs |
| `annotations.registry.register_format`, loader/dumper/spec getters and format lists | Register/query format adapters and FormatSpec | Process-local registry changes; no scanner creation |
| `annotations.validation.validate_annotation` | Validate an existing Annotation with optional supplied image dimensions | No file reads; does not discover dimensions |
| `compare_class_schemas` | Compare ordered class lists before combining data | Strict comparison, not conversion remapping |
| `discover_parts/select_annotation_source` | Discover supported fixed layouts and choose one annotation candidate | No annotation parsing or pixel checks; no arbitrary recursive layout guessing |
| `open_dataset/DatasetCollection.open` | Complete discovery/explicit-source scanning and collection creation | Composition entry points, not a compulsory pipeline for other functions |
| `partition_paths/unique_paths` | Sort/deduplicate/allocate explicit paths | No Profile or file content required; optional normalization differs from lexical identity |
| `validate_partition_members/combine_partitions` | Validate path overlap or combine existing path partitions | No Collection exclusion history or permanent stem filtering |
| `save_partition_txts` | Write explicit member paths | No image moves or repartitioning |
| `DatasetCollection.split` | Repartition current effective members while inheriting exclusions | Requires a collection result; pool duplicate stems before allocation, no source reads |
| `tools.convert_dataset/plot_dataset/export_dataset` | Batch processing using either Profile type | Requires the result contract, not a live Dataset; conversion/rendering may read dimensions/pixels and write outputs |
| `tools.convert_dirs/plot_dirs/export_dirs` | Temporary explicit scanning followed by batch processing | Convenience composition, no second conversion/rendering implementation |
| `visualize.plot.plot` | Draw one supplied image/Annotation or load explicitly supplied paths | No Profile required; use Plotter directly for array drawing |
| `detection.prediction_from_records/target_from_records/class_id_from_record` | Convert records to numeric predictions/GT or resolve a class ID | No model, file or Profile dependency; no box matching |
| `metrics.detection_metrics` | EvaluationSample list to DetectionMetrics | Bypasses mapping adaptation; caller owns sample pairing and coordinate/class agreement |
| `metrics.box_iou/match_predictions/compute_ap/ap_per_class` | Direct array-level matching and metric calculations | Preconditions still apply: match_predictions expects confidence-sorted predictions |
| `metrics.normalize_iou_thresholds/expand_iou_thresholds/ap_keys_from_iou/smooth` | Threshold normalization, keys and curve smoothing | Pure math; threshold deduplication is not prediction-box NMS |
| `inventory_dataset/content_digest/file_hashes` | File inventories and content SHA256 | Reads file contents; does not certify sample validity or update Profiles |
| `compare_file_hashes/fingerprint_identity` | Compare existing hashes or hash caller-defined identity data | No implicit monitoring, migration, discovery or scan |
| `utils.image.read_image_info/resolve_image_wh` | Obtain dimensions when a consumer needs them | Header reads on demand; explicit supplied dimensions can avoid reading |
| `utils.image.validate_image` | Check one image, optionally decode=True | Independent of Dataset; no automatic mutation of Profiles or exclusions |
| `utils.image.iter_image_paths/get_image_paths/register_reader/image_to_base64` | Enumerate direct image files, extend metadata readers, or encode image content | Enumeration is not content validation; Base64 reads the file |
| `utils.file.load_txt/save_txt/merge_txt_files/replace_text_in_txt` | Explicit TXT I/O, merging and replacement | Replacement may write back; merge output cannot refer to an input; no membership generation |
| `utils.file.find_parent_dirs_with_subfolder/build_stem_index/validate_source_tree` | Directory search, format-priority stem indexing and structural source-tree checks | Not arbitrary layout discovery; priority selection differs from permanent duplicate-stem exclusion |
| `utils.file.ensure_dir/flatten_dir/rm_empty_dirs` | Create directories, move files into a flat directory, or remove empty directories | Explicit filesystem changes, never implicit Dataset scan steps |

Read-only source guarantees apply to scanning and collection organization, not every
utility in the package. Output writing and explicit file organization have their own
side effects. Underscore-prefixed builders/modules and _ProfileStats/_ProfileAccess
are internal implementation, not independent public entry points.

## Relationship to PyTorch Dataset and DataLoader

PyTorch's map-style Dataset provides indexed samples; IterableDataset provides a
sample stream. They define access rather than dsetkit's scan-result contract. See
[PyTorch's Dataset tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

| Layer | Main role |
| --- | --- |
| dsetkit Dataset / Collection / Profiles | Scan/check sources, organize effective members and persist analysis/exclusions |
| Caller-owned PyTorch Dataset adapter | Read chosen effective samples, decode and transform them into model inputs |
| PyTorch DataLoader | Arrange sampling, batching, collation and worker loading |

DataLoader supports prefetching and memory pinning. Its loading order is distinct
from a persisted train/val/test split. See the [official DataLoader reference](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

The intended integration is source files → dsetkit Profiles/optional Collection →
a caller-owned PyTorch Dataset adapter → DataLoader → model. No PyTorch adapter is
currently included. dsetkit.Dataset does not implement the training sample indexing
protocol; indexing a Profile returns DatasetSample path metadata. It does not itself
provide decoded tensors or detection-batch collation. The adapter chooses a partition,
applies consistent image/box transforms and aligns category/coordinate conventions.
YOLO GT is normalized xyxy; VOC/LabelMe GT is pixel xyxy. Variable-length detection
targets require a suitable collation strategy. The scanner need not survive training.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `dataset.py` | Explicit single-source scanning, file checks and val() GT adaptation |
| `collection.py` | Discovery/scan orchestration, collection lifetime and split() |
| `collection_profile.py` | Source references, immutable membership, cumulative exclusions and collection/partition statistics |
| `profile.py` | Complete immutable results, derived statistics and processing access |
| `_profile_io.py` | Profile JSON encoding/decoding and structural checks; no source I/O |
| `annotations/validation.py` | Detection geometry rules independent of files and serialization |
| `annotations/io.py` and `formats/` | Annotation file I/O and format adapters |
| `annotations/_mapping.py` | Class mapping checks and conversion mapping |
| `discovery.py` | Fixed-depth layout discovery and annotation-source selection |
| `inventory.py` | Raw file inventories and explicit content fingerprints |
| `partitions.py` | Path membership, train/val/test partition policy and explicit TXT output |
| `tools.py` | Batch conversion, visualization and export orchestration over Profile |
| `evaluator.py` / `metrics.py` | Pure sample pairing and validation / detection metric calculation; no file or model I/O |

## One partition policy

`partition_paths()` compares paths lexically, deduplicates, sorts by string and
shuffles with an independent Python `random.Random(seed)`. Exactly three finite
nonnegative ratios must sum to one within 1e-6, with positive train. Train and
validation counts are floored; test receives the remainder. With zero test,
validation receives the remainder instead. Zero validation is supported. Empty
train is rejected. Alias resolution is the caller's responsibility for this
operation; saved lists are never read or rewritten implicitly.

`combine_partitions()` resolves paths by default, deduplicates within each split,
sorts by string and rejects train/val/test overlaps. It never reassigns samples.
Pass `normalize=False` for lexical identity. `validate_partition_members()` also
rejects same-split duplicates. `PartitionError` exposes kind/left/right/count for
application messages. These operations do not deduplicate by image content.

`save_partition_txts()` writes explicit groups without changing order or
repartitioning. Obtain a Profile with `open_dataset().profile()`, pass `profile.image_paths`
to `partition_paths()`, then save the groups explicitly. There are no additional
split-named wrappers in `dsetkit.tools`.

## Inventories and fingerprints

`dsetkit.inventory.inventory_dataset(root, source_format=None)` uses the same
discovery results to return native-layout status, all/train/val/test members and
SHA256 hashes. It includes only supported images and matching-format annotation
files, including orphan labels. An omitted format includes every discovered
source. Neither discovery nor inventories write source files.

`file_hashes(paths, relative_to=None)` hashes explicit files, using absolute keys
by default. Explicit root-relative keys allow relocation and reject paths outside
the root. `compare_file_hashes()` reports added, removed and changed keys.
`fingerprint_identity()` hashes sorted-key JSON with default Python JSON encoding;
callers own the identity schema and path policy. There is no implicit migration
of historical fingerprints. Inventories observe live files rather than atomic
filesystem snapshots.

## Breaking changes

- `export_scan()`, `from_scan()` and `DatasetView` are removed. Persist complete
  Profiles with `profile.export()` / `DatasetProfile.from_json(text)`. There is no
  automatic legacy-state restoration or Dataset reconstruction from Profile.
- Conversion, plotting and export accept Profile only. Read sample fields and
  statistics from Profile; scanners do not forward sample iteration, annotations or
  path access. Dataset-level evaluation is provided explicitly by `val()`.
- `Evaluator(names=None).evaluate(predictions, ground_truths)` replaces the
  Profile-based evaluator. Prediction loading hooks and terminal printing are removed.
  `EvaluationSample.sample_id` replaces `image_id`; IDs are opaque nonblank strings.
- Dataset.check_images()/check_annotations() update its current Profile and return the new result. Profile/Collection checks and DatasetProfile.combine() are removed; Collection accepts Profiles, not Dataset instances. Restore sources with DatasetProfile.from_json() and collections with DatasetCollectionProfile.from_json().


- Image augmentation is removed. `dsetkit.augment` and the `flip_dataset`,
  `flip_dirs`, `rotate_dataset`, and `rotate_dirs` functions in `dsetkit.tools`
  are no longer available, including schema-level flip and rotation helpers.

- `tools.split_dataset()` and `tools.split_dirs()` are removed. Use the explicit
  `open_dataset()` → `partition_paths()` workflow, then iterate over the dictionary
  and write each partition to TXT as shown in the example.

- `dsetkit.split` and its `split_paths`, `split_tvt`, `save_split_txts` entry points
  are removed. Use `partition_paths` and explicit persistence. Short one/two-ratio
  forms are removed; for example `[0.8]` becomes `(0.8, 0.2, 0)`.
- `dsetkit.layout` is removed. Discovery lives in `discovery.py`; scanning in
  `dataset.py` and analysis results in `profile.py`. Import recommended APIs from `dsetkit`.
- The flat-only `discover_dataset()` entry point is removed. Use `discover_parts()`
  and `select_annotation_source()`. Selected annotation fields on discovery
  results are replaced by one canonical candidate collection.
- New partition jobs use the unified sorting/deduplication/empty-train policy.
  Persisted partition lists retain their previous contents.
- Inventories exclude direct split-root files, sidecars and mismatched formats.
  Applications with older inventories must rescan and create new versions when
  observed membership/hashes differ; old inventories are not rewritten.

Applications retain database records, snapshot IDs/history, lineage, training
inputs, workspace paths and lifecycle management.

## Scan options and processing membership

`Dataset`, `open_dataset()` and dataset-building `*_dirs` tools require
`task_type="detection"`; other tasks are rejected. `Dataset` and `open_dataset()`
define independent options: `check_annotations=True`, `check_images=False`.
Path stem conflicts are excluded before label parsing and image inspection.
Remaining selected annotations are parsed once. The annotation check validates boxes;
image checking additionally audits containers, fully decodes images and uses actual
image dimensions for box bounds, overriding annotation size metadata.
Disabling a check assumes processable content; it does not certify that content.
Unparseable annotations and unsupported detection structures remain errors.

`names` is optional. VOC/LabelMe infer names in first-occurrence order from
processable samples. YOLO preserves actual IDs; missing names
appear in `profile.unnamed_class_ids` and `warnings` without rejecting samples.
`class_ids`, nullable `class_names` and `class_instances` are parallel sequences.
Conversions involving YOLO require a complete final name mapping. Single-file conversion requires explicit names; dataset conversion uses profile.names only when names is None.
VOC/LabelMe to YOLO maps category names to positions in the final list. All conversion entry points reject identical source and target formats after lowercase normalization.
Mixed YOLO/named-format collections require a shared explicit mapping.
`source_format` accepts real formats or None, never `"unannotated"`; a part without
an annotation directory has format None.

Dataset.check_images() checks known raw paths without rediscovery or label reads;
Dataset.check_annotations() checks cached annotations. Both preserve prior errors and
replace only the scanner's current result. Profile results never perform checks.
Both Profile types provide export()/from_json() for full persistence.
Iteration, counts and processing reuse this membership and cached annotations;
pixel-based operations may still read the target image. Errors never become backgrounds.
`raw_samples` retains paths only; `invalid_samples` and `profile.errors` expose
exclusions. `total_images`/`total_label_files` count raw entries.
Sources are assumed immutable; an explicit build rescans them. Structural inspection
rejects symbolic links, special files and line breaks; duplicate stems are image
errors detected before content checks. Sample discovery remains fixed-depth. `inventory_dataset()` remains a task-independent raw inventory.

## Evaluation inputs and IoU thresholds

`Evaluator(names=None).evaluate(predictions, ground_truths)` is a pure numerical
comparison of existing detections and explicit GT. Both inputs are mappings keyed
by unique nonblank string `sample_id`. IDs have no path semantics, require no suffix
and never trigger file access. Values can be `PredictionResult` / `AnnotationTarget`
or record sequences containing `bbox` (xyxy), `class_id`/`cls` or a named `label`,
and `conf` for predictions. Numeric IDs do not require names; string labels do.
Prediction and GT must share the same coordinate system and category mapping.
Normalized coordinates are supported without image sizes.

GT defines the sample population. Missing predictions are empty; prediction IDs
absent from GT are rejected even when their prediction lists are empty. Explicit
empty GT entries represent backgrounds and retain their predictions for false-positive
calculation. Results are returned without printing; `images` counts all GT samples.
AP continues to average classes present in GT, following the existing fixed
Ultralytics reference covered by regression fixtures.

Profile never calls Evaluator. Callers can explicitly adapt its cached GT, choosing
IDs that remain unique across partitions:

```python
from dsetkit.evaluator import Evaluator

ground_truths = {
    profile.output_path(sample).with_suffix("").as_posix(): profile.ground_truth(sample)
    for sample in profile
}
# An empty prediction mapping evaluates all GT samples as missed detections.
metrics = Evaluator(names=profile.names).evaluate({}, ground_truths)
```

Reading annotations, extracting GT and converting coordinates are caller-side
operations. Standalone annotation results can be evaluated without creating a Profile
or scanning images. Presentation belongs to the caller, CLI or UI.

`Dataset.val(predictions, *, conf_threshold=0.001, iou=0.5)` and
`DatasetCollection.val(...)` adapt the current Profile to the pure Evaluator and return
its metrics dictionary. Dataset must be built first; open_dataset() already completes
scanning and currently always returns DatasetCollection. Both dataset types expose
val(); Profile does not. No model is loaded and no metrics are printed.

Single-source IDs are stems. Collections use the suffix-free POSIX rendering of
output_path: train/sample for a unique split, dataset_1/sample for multiple unsplit
sources, and dataset_1/train/sample for repeated splits. One unsplit source uses bare
stems. Predictions use the collection's class IDs and the coordinate convention of
each sample's GT: normalized xyxy for YOLO, pixel xyxy for VOC/LabelMe. This convention
is unchanged by image checking or cached dimensions. Mixed-format collections can
have different coordinate conventions per sample; each prediction/GT pair must agree.
Only effective samples enter GT. val() never reads images or source labels, rescans,
checks files or mutates results.

`metrics` validates finite IoU thresholds in `(0, 1]`, sorts them and removes duplicates
before AP calculation. One normalized threshold below 0.95 expands in 0.05 steps up to
but not beyond 0.95; multiple thresholds are used directly. Results include AP at the
lowest threshold and mean AP over the resulting thresholds. One resulting threshold
produces only its point metric, such as `mAP95` or `mAP100`.

LabelMe `imagePath` must be a string. Missing or incorrectly typed values are recorded
as annotation parse errors. Profile JSON requires absolute source and sample paths and
rejects non-finite numeric tokens, including exponent overflow. Invalid cached VOC box
coordinates retain their explicit string encoding for lossless error-state restoration.
