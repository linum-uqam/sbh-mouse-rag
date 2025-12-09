# SBH-MOUSE-RAG

## Setup environnement

```bash
python3.11 -m venv .venv; source .venv/bin/activate;
pip install uv
uv sync 
```

## Volume
```bash
python -m scripts.volume_usage

```

## Dataset
```bash
python -m scripts.create_dataset
```

## Index
```bash
python -m scripts.create_index
python -m scripts.create_index --k-normals 16 --slice-size 512 --patch-scales 1 2 4 8 --patch-overlap 0.5 --out-dir dino_16_1-2-4-8_05
```

## Search

**With a whole slice**
```bash
# Base search (only faiss cosine similarity)
python -m scripts.search_index out/dataset/data/00044_a.png --k 10 --save-dir out/search/00044_a

# With re-ranker (using MLP on embedding as reranker)
python -m scripts.search_index out/dataset/data/00044_a.png \
  --save-dir out/search/00044_a_rerank \
  --k 50 \
  --use-reranker \
  --rerank-topk 50
```

**With a crop slice**
```bash
python -m scripts.search_index index/test/test.PNG --k 10 --save-dir out/search/test
python -m scripts.search_index index/test/test.PNG \
  --save-dir out/search/test_rerank \
  --k 50 \
  --use-reranker \
  --rerank-topk 50
```

## Eval
**Run searches**
```bash
# Baseline
python -m scripts.run_eval \
  --csv out/dataset/dataset.csv \
  --source both \
  --final-k 10 \
  --k-per-angle 64 \
  --save-dir out/eval/base \
  --save-k 3

# With reranker
python -m scripts.run_eval \
  --csv out/dataset/dataset.csv \
  --source both \
  --final-k 100 \
  --k-per-angle 64 \
  --save-dir out/eval/rerank \
  --use-reranker \
  --rerank-topk 100 \
  --reranker-model-path out/reranker/reranker.pt \
  --reranker-device cuda \
  --save-k 10
```

**Run the report script**
```bash
python -m scripts.run_report --csv out/eval/eval_hits.csv
python -m scripts.run_report --csv out/eval/rerank/eval_hits.csv
```

**Train re-ranker**
```bash
python -m scripts.train_reranker \
    --data-mode volume \
    --allen-cache-dir volume/data/allen \
    --allen-resolution 25 \
    --real-nifti volume/data/real/registered_brain_25um.nii.gz \
    --n-samples 50000 \
    --slice-size 224 \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```


## Paper 

**Install Latex locally (1.1Go)**
```bash
sudo apt update;
sudo apt install \
  texlive-publishers \
  texlive-science \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-lang-french \
  texlive-lang-english \
  latexmk;
```

Then, on vscode, go to extensions and install [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop). 

**Use LaTex extension**

Build the pdf with `Ctrl+Shift+P`, or click on the Tex logo (left side bar) and then `View LaTex PDF` > `View in VSCode tab`.


## Registering (optionnal)

```bash
python scripts/register/resample_real_to_25um.py;
python scripts/register/find_best_orientation.py \
  --allen volume/data/allen/average_template_25.nrrd \
  --real  volume/data/real/registered_brain_25um.nii.gz \
  --out-real volume/data/real/registered_brain_25um_preoriented.nii.gz \
  --subsample 4
python scripts/register/view_registration_napari.py
```