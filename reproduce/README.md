# Reproduction workflow

## 1. Verify the frozen reference package

```bash
python tools/verify_reference_hashes.py
python tools/export_reference_summaries.py
```

## 2. Add the original simulator

Copy `dlms_experiment3_GLM2_3.py` into `src/`.

## 3. Install minimal Python dependencies

```bash
python -m pip install -r environment/requirements-minimal.txt
```

## 4. Run the original experiment program

```bash
python src/dlms_experiment3_GLM2_3.py
```

The original program writes generated files to `code_results/`. Do not overwrite `results/reference/`; use it as the frozen comparison target.
