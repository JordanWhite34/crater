# Dataset layout

Datasets are local, ignored artifacts. Keep each domain and learning task
separate:

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
      images/{train,val,test}/
      annotations/
      manifests/
    damage/
      images/{train,val,test}/
      labels.csv
```

Detection annotations use COCO bounding boxes. Damage data contains component
crops and classification labels; it must not be mixed into detector annotation
files.

Create the reproducible civilian detector data with:

```powershell
python tools\data\download_carparts23.py
```

Military imagery and all damage imagery must include source/license metadata.
Split source images, scenes, vehicles, and sequences before generating crops;
every crop inherits its source image's split. This prevents crops from the same
vehicle or source asset leaking across train, validation, and test sets.
