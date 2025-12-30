# FragEvo: A Molecular Language Model Driven Genetic Algorithm for Multi-Objective Molecular Generation

This repository implements an end-to-end workflow for drug-like molecule optimization: a generative model + genetic algorithm (GA) + docking-based evaluation + selection. Under a target receptor constraint, the system iteratively evolves a population and keeps better molecules.

---

## 1. Reproducibility (setup + run)

### 1.1 Environment & dependencies

#### Python environment (recommended: Conda)
Use the provided environment file: `fragevo.yml`

```bash
conda env create -f fragevo.yml
conda activate fragevo
```

Optionally ensure these are installed (some environments may miss them):
```bash
pip install -U psutil tqdm openpyxl
```

#### AutoGrow dependency (external)
The `operations/` pipeline expects an `autogrow/` package at the repo root (`./autogrow`). You can obtain it from either:

- Upstream AutoGrow4.0 (original project): [`durrantlab/autogrow4`](https://github.com/durrantlab/autogrow4/tree/master/autogrow)
- FragEvo-adapted fork (minor modifications): [`tg929/autogrow`](https://github.com/tg929/autogrow)

Place it at `./autogrow` (download, clone, symlink, or git submodule), for example:
```bash
git clone https://github.com/tg929/autogrow autogrow
```

#### Docking toolchain
Docking relies on:
- MGLTools: `mgltools_x86_64Linux2_1.5.6/`
- AutoDock Vina / QVina2 executables: `autogrow/docking/docking_executables/...`
- OpenBabel (installed via Conda)

##### Installing MGLTools (1.5.6)
Download MGLTools from the official website ([MGLTools downloads](https://ccsb.scripps.edu/mgltools/downloads/)), then install it locally:

```bash
tar -zxvf <mgltools-*.tar.gz>
cd mgltools_x86_64Linux2_1.5.6
./install.sh
cd ..
```

Keep the installed directory at `./mgltools_x86_64Linux2_1.5.6`, or update the corresponding paths in your config (e.g. `docking.mgltools_dir`, `docking.mgl_python`, `docking.prepare_receptor4.py`, `docking.prepare_ligand4.py` in `fragevo/*.json`).

If you see `Permission denied` / `Exec format error`, ensure the docking binary is executable, e.g.:
```bash
chmod +x autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina
```

#### GPU (optional)
FragMLM generation can use GPU; it falls back to CPU if CUDA is unavailable (much slower).

The workflow does not explicitly pass `--device` to `fragmlm/generate_all.py`; the simplest way to choose a GPU is:
```bash
export CUDA_VISIBLE_DEVICES=0
```

### 1.2 Data & configuration

#### Initial population (SMILES)
Format: one SMILES per line (extra columns are allowed; the first column must be the SMILES).

Example files shipped in this repo:
- `datasets/initial_population/my_initial_population.smi`
- `datasets/source_compounds/naphthalene_smiles.smi`

By default, both `fragevo/config_example.json` and `fragevo/config_fragevo.json` point to `datasets/initial_population/my_initial_population.smi`. To use a different initial population, edit `workflow.initial_population_file` in the config.

#### Receptors and docking boxes
Receptor configuration is under `receptors` in the JSON config:
- `default_receptor`: used when `--receptor` is not provided
- `target_list`: iterated by `--all_receptors`

Each receptor entry needs:
- `file`: receptor PDB/PDBQT path (relative to project root)
- `center_x/y/z` and `size_x/y/z`: docking box parameters

#### Switching selection strategies (single / multi / CompScore)
For the GA baseline (`fragevo/config_example.json`) and the standard FragEvo pipeline (`fragevo/config_fragevo.json`), switch selection by editing:
- `selection.selection_mode`:
  - `single_objective`: docking-score only (see `selection.single_objective_settings`)
  - `multi_objective`: NSGA-II multi-objective (see `selection.multi_objective_settings`)

CompScore (RAG-score) selection is provided as a separate runnable config + entrypoint:
- Config: `fragevo/config_fragevo_rag.json` (`selection.selection_mode = "rag_score"` + `selection.rag_score_settings`)
- Entry: `FragEvo_rag.py` (the standard executors only handle `single_objective` / `multi_objective`)

### 1.3 How to run (reproducible commands)

All commands below assume you are at the repo root.

#### Pure GA baseline

1) Single receptor (default receptor from the config)
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --output_dir GA_output_demo
```

2) A specific receptor (the name must exist in `config_example.json -> receptors.target_list`)
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --receptor 4r6e --output_dir GA_output_demo
```

3) All receptors (`target_list`)
```bash
python fragevo/GA_main.py --config fragevo/config_example.json --all_receptors --output_dir GA_output_all
```

#### FragEvo hybrid pipeline (FragMLM + GA)

1) Single receptor
```bash
python FragEvo_main.py --config fragevo/config_fragevo.json --receptor parp1 --output_dir FragEvo_output_demo
```

2) All receptors (parallelism controlled by `performance` in the config)
```bash
python FragEvo_main.py --config fragevo/config_fragevo.json --all_receptors --output_dir FragEvo_output_all
```

#### CompScore (RAG-score) selection (optional)
Entry: `FragEvo_rag.py`  
This pipeline reuses the FragEvo workflow and only swaps the selection stage to `operations/selecting/selecting_rag_score.py`.

Config: `fragevo/config_fragevo_rag.json` (includes optional docking-score elitism via `selection.rag_score_settings.elitism`).  
Note: this runnable CompScore pipeline does **not** depend on `GA_gpt_rag/` (if the folder exists in your workspace, treat it as a reference snapshot only).

```bash
python FragEvo_rag.py --config fragevo/config_fragevo_rag.json --receptor parp1 --output_dir FragEvo_output_rag
```

### 1.4 Outputs & aggregation

Example (`--output_dir FragEvo_output_demo`, `--receptor parp1`):

```
FragEvo_output_demo/
  parp1/
    execution_config_snapshot.json
    chem_metric_cache.json                  # for multi-objective selection (QED/SA cache)
    generation_0/
      initial_population_docked.smi
    generation_1/
      current_parent_smiles.smi
      masked_fragments.smi
      gpt_generated/gpt_generated_molecules.smi
      crossover_filtered.smi
      mutation_filtered.smi
      offspring_docked.smi
      generation_1_evaluation.txt
    ...
```

The per-generation evaluation report (e.g. `generation_10_evaluation.txt`) is produced by `operations/scoring/scoring_demo.py`, and includes:
- Docking score: Top1 / Top10 mean / Top100 mean
- Novelty: novelty vs. the initial population
- Diversity: diversity of Top100
- QED / SA: Top100 mean values

For an output directory containing multiple receptor subfolders (e.g. `FragEvo_output_all/`), you can aggregate into an Excel file:

```bash
python operations/stating/statistics_output_demo.py --output_dir FragEvo_output_all --excel_output all_statistics.xlsx
```

---

## 2. High-level idea (three key modules)

### 2.1 FragMLM (`fragmlm/`)
**Role: propose novel candidates to escape local optima.** Each generation, parent molecules are decomposed into fragment sequences, a suffix is masked (we keep only a fragment prefix as a condition), GPT continues the sequence at the fragment level, and fragments are reconstructed back to full-molecule SMILES.

- Decomposition + masking: `datasets/decompose/demo_frags.py`
  - Typical line format: `[BOS]frag1[SEP]frag2[SEP]...[SEP]`
  - Supports **dynamic masking**: mask more fragments early (exploration), fewer late (exploitation/refinement)
- Batch generation entry: `fragmlm/generate_all.py`
  - Default checkpoint: `fragmlm/weights/dpo_0_400.pt`
  - Output: `*.smi` (one generated SMILES per line)

### 2.2 GA (Genetic Algorithm, `operations/`)
**Role: controlled local search around existing molecules.** The project uses AutoGrow-style operators for crossover/mutation, optional filtering, and docking evaluation.

- Crossover: `operations/crossover/crossover_demo_finetune.py`
- Mutation: `operations/mutation/mutation_demo_finetune.py`
- Filtering: `operations/filter/filter_demo.py`
- Docking evaluation: `operations/docking/docking_demo_finetune.py`

### 2.3 Selection (`operations/selecting/`)
**Role: select next-generation parents from a merged pool of “current parents + offspring (from GPT/GA)”.** Three selection families are provided:

1) **Single-objective (docking score only)**: `operations/selecting/molecular_selection.py`  
   - Rank / Roulette / Tournament selectors  
2) **Multi-objective (NSGA-II)**: `operations/selecting/selecting_multi_demo.py`  
   - Default objectives: minimize Docking, maximize QED, minimize SA  
   - Metric caching: `utils/chem_metrics.py` (cache persisted under the run root)  
3) **RAG-score (composite score)**: `operations/selecting/selecting_rag_score.py`  
   - `y = DS_hat * QED * SA_hat` (see script header)  
   - Workflow entry: `FragEvo_rag.py`

---

## 3. End-to-end workflow

Two main pipelines are included:

### 3.1 Pure GA pipeline (baseline)
- Entry: `fragevo/GA_main.py`
- Core executor: `operations/operations_execute_demo.py`

Per-generation (conceptually):
1. Gen0: deduplicate initial population + dock
2. Extract parent SMILES
3. Crossover + filter
4. Mutation + filter
5. Dock offspring
6. Selection: pick next parents from merged pool (parents + offspring)
7. Evaluate the selected parents and write a report (Top1/Top10/Top100/Novelty/Diversity/QED/SA)

### 3.2 FragEvo hybrid pipeline (FragMLM + GA)
- Entry: `FragEvo_main.py`
- Core executor: `operations/operations_execute_fragevo_demo.py`

Difference vs. GA baseline: insert **decompose+mask → GPT generation** before GA, then feed GPT outputs into the GA input pool:

```
Parents (plain SMILES)
  └─ Decompose & mask (datasets/decompose/demo_frags.py)
      └─ GPT generation (fragmlm/generate_all.py)
          └─ GA input pool = parents + GPT
              └─ crossover/mutation → docking → selection → next parents
```

---

## 4. Code map (where to start reading)

If you want to understand the full pipeline quickly, start here:

- Entrypoints
  - `FragEvo_main.py`: hybrid pipeline (supports multi-receptor serial/parallel)
  - `fragevo/GA_main.py`: pure GA baseline
  - `FragEvo_rag.py`: FragEvo with RAG-score selection (selection stage only)
- The three core modules
  - FragMLM: `fragmlm/generate_all.py`
  - GA operators: `operations/crossover/*`, `operations/mutation/*`, `operations/filter/*`
  - Selection: `operations/selecting/*`
- Docking
  - `operations/docking/docking_demo_finetune.py`
  - Receptor/box settings: `receptors` blocks in `fragevo/*.json`
- Metrics & reporting
  - Per-generation report: `operations/scoring/scoring_demo.py`
  - Multi-receptor aggregation: `operations/stating/statistics_output_demo.py`

---

## 5. Reproduction tips (recommended)

1) **Start with a small smoke test**: reduce `max_generations`, `number_of_crossovers`, `number_of_mutants`, and `n_select` to validate the full chain (generation → docking → selection → reporting).
2) **Docking is the slowest stage**: for fast validation, reduce `docking_exhaustiveness` and `docking_num_modes`.
3) **Parallelism**
   - Across receptors: `performance.parallel_processing` + `performance.max_workers`
   - Within a receptor (docking etc.): `performance.number_of_processors` (`-1` means auto)

---
<!-- 
## 6. Troubleshooting

- **Docking fails / empty outputs**: check receptor paths and box parameters first; then check the Vina/QVina2 binary permissions.
- **MGLTools errors**: ensure `mgltools_x86_64Linux2_1.5.6/bin/pythonsh` exists and is executable.
- **GA fails due to missing initial population**: verify `workflow.initial_population_file` points to an existing `.smi` file.
- **FragMLM generation is slow**: verify CUDA-enabled PyTorch, or select a GPU via `CUDA_VISIBLE_DEVICES`. -->
