# Dataset layout

Datasets are ignored by default. Keep each domain and learning task separate.
The Humvee source comparison export is an explicit versioned exception: its
images use Git LFS, while its COCO JSON, integrity manifest, and documentation
use normal Git.

```text
datasets/
  civilian/
    carparts23/
      images/{train,val,test}/
      annotations/
      manifests/
      audit/
    damage/
      images/{train,val,test}/
      labels.csv
  military/
    components/
      README.md
      source/
        humvee/                 # immutable source export; not training-ready
          instances_default.json
          source_manifest.csv   # tracked integrity/provenance inventory
          commons_*.{jpg,png}
      images/{train,val,test}/
      annotations/             # task-specific prepared COCO split files
      manifests/
    damage/
      images/{train,val,test}/
      labels.csv
```

Detection annotations use COCO bounding boxes. Damage data contains component
crops and classification labels; it must not be mixed into detector annotation
files.

Keep received datasets intact under `source/`. Only deliberately approved
source datasets should be exempted from the default ignore rule. A preparation
step must validate provenance, establish the experiment taxonomy, assign
grouped train/validation/test splits, and write the prepared `images/`,
`annotations/`, and `manifests/` artifacts. Training experiments consume only
those prepared artifacts, never a source export directly.

Create the reproducible civilian detector data with:

```powershell
python tools\data\download_carparts23.py
```

Military imagery and all damage imagery must include source/license metadata.
Split source images, scenes, vehicles, and sequences before generating crops;
every crop inherits its source image's split. This prevents crops from the same
vehicle or source asset leaking across train, validation, and test sets.
