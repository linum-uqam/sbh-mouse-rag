# SBH-MOUSE-RAG

## Setup environnement

```bash
python3.11 -m venv .venv; source .venv/bin/activate;
pip install uv
uv sync 
```

## Volume
```bash
python -m volume.usage
```

## Dataset
```bash
python -m scripts.create_dataset
```

## Index
```bash
python -m scripts.create_index
python -m scripts.create_index --k-normals 16 --slice-size 512 --patch-scales 1 2 4 8 --patch-overlap 0.5
```

## Search

**With a whole slice**
```bash
python -m scripts.search_index dataset/data/13_a_slice.png --k 10 --save-dir results/13_a_slice
```

**With a crop slice**
```bash
python -m scripts.search_index index/test/test.PNG --k 10 --save-dir results/test
```

## Eval
```bash
python -m scripts.run_eval \
  --csv dataset/dataset.csv \
  --source allen \
  --limit 100 \
  --save-dir eval/out \
  --save-k 10
```


