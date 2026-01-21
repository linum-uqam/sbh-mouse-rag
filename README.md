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
python -m scripts.create_index # --k-normals 16 --slice-size 512 --patch-scales 1 2 4 --patch-overlap 0.25
```

## Search

**With a whole slice**
```bash
# Base search (only faiss cosine similarity)
python -m scripts.search_index \
  out/dataset/data/00005_a_crop1.png \
  --k 20 \
  --k-per-angle 64 \
  --save-dir out/search/baseline
```

## Eval
**Run eval**
```bash
# Baseline
python -m scripts.run_eval \
  --csv out/dataset/dataset.csv \
  --source both \
  --final-k 100 \
  --k-per-angle 64 \
  --save-dir out/eval/base \
  --save-k 3 \
  --distance-grid 32 \
  --distance-trim 0.05
```

**Run the report script**
```bash
python -m scripts.run_report --csv out/eval/base/eval_hits.csv
```

## Re-ranker

**Build a training dataset**
```bash
python -m scripts.create_reranker_dataset \
  --num-slices 1000 \
  --out-dir out/reranker_dataset/data \
  --csv-path out/reranker_dataset/dataset.csv \
  --seed 123
```

**Run the eval to get the hits**
```bash
# Run the eval to create the base hits that will be use to train the reranker.
python -m scripts.run_eval \
  --csv out/reranker_dataset/dataset.csv \
  --source both \
  --final-k 100 \
  --k-per-angle 64 \
  --save-dir out/reranker_dataset \
  --distance-grid 32 \
  --distance-trim 0.05
```

**Train the re-ranker**
```bash
python -m scripts.train_reranker \
  --hits-csv out/reranker_dataset/eval_hits.csv \
  --dataset-csv out/reranker_dataset/dataset.csv \
  --out out/reranker/reranker_listwise.pt \
  --train-topk 100 \
  --list-k 100 \
  --epochs 100 \
  --batch-size 64
```

**Search with the reranker**
```bash
# With re-ranker (using MLP on embeddings as reranker)
python -m scripts.search_index \
  out/dataset/data/00005_a_crop1.png \
  --k 10 \
  --k-per-angle 64 \
  --save-dir out/search/reranked \
  --use-reranker \
  --reranker-model out/reranker/reranker_listwise.pt \
  --rerank-topk 100
```

**Eval the reranker**
```bash
# With reranker
python -m scripts.run_eval \
  --csv out/dataset/dataset.csv \
  --source both \
  --final-k 100 \
  --k-per-angle 64 \
  --save-dir out/eval/rerank \
  --save-k 3 \
  --distance-grid 32 \
  --distance-trim 0.05 \
  --use-reranker \
  --rerank-topk 100 \
  --reranker-model-path out/reranker/reranker_listwise.pt \
  --reranker-device cuda \
  --reranker-batch-size 256
```

**Run the report for the reranker results**
```bash
python -m scripts.run_report \
  --baseline out/eval/base/eval_hits.csv \
  --rerank out/eval/rerank/eval_hits.csv

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