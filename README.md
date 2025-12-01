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
python -m scripts.search_index out/dataset/data/00044_a.png --k 10 --save-dir out/search/00044_a
```

**With a crop slice**
```bash
python -m scripts.search_index index/test/test.PNG --k 10 --save-dir out/search/test
```

## Eval
**Run searches**
```bash
python -m scripts.run_eval   --csv out/dataset/dataset.csv   --source both   --save-dir out/eval/   --save-k 10 --final-k 100
```

**Run the report script**
```bash
python -m scripts.run_report --csv out/eval/eval_hits.csv
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