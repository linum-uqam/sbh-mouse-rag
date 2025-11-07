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
python -m dataset.create
```

## Index
```bash
python -m index.create
```

## Search

**With a whole slice**
```bash
python -m index.search dataset/data/67_a_slice.png --mode col --angles 0 90 180 270 --scales 1 2 4 8 14 --base-topk-for-col 400
```

**With a crop slice**
```bash
python -m index.search index/test/test.PNG --mode col --angles 0 90 180 270 --scales 1 2 4 8 14 --base-topk-for-col 400
```

## Eval
```bash
todo
```

