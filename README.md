# ring-finger-semseg
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -r requirements/main.txt
(.venv) $ pip install -r requirements/dev.txt
```

## 教師データの生成
```
python generate_data.py \
  --background_image_path ../fiftyone/coco2017/validation/data \
  --blender_image_base_path ../blender-for-finger-segmentation/data2 \
  --target training
```

## 推論
```
python inf_segformer.py  models/segformer/ logits_mit_b1.pt
```

## Visualize
```
PYTHONPATH=. python scripts/visualize.py outputs/logits_mit_b4.pt [OUTPUT_PNG_FILENAME]
```
