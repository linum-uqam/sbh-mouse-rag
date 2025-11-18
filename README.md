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
python -m index.search dataset/data/13_a_slice.png --mode col --debug --save-dir index/out/ 
```

**With a crop slice**
```bash
python -m index.search index/test/test.PNG --mode col --debug --save-dir index/out/ 
```

## Eval
```bash
python -m eval.run \
  --mode col \
  --save-dir eval/out \
  --final-k 10 \
  --limit 50
  --scales 1 8 14
  --token-scales 8 14

```


