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
  --background_image_path=../coco2017/train/data
```
