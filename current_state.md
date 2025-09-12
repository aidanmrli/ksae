### Summary
- Implemented LISTA forward pass in `ksae/src/ksae/lista/encoder.py` to run exactly T iterations and to optionally return intermediates `B`, `C^(t)`, `Z^(t)`.
- Added tests in `ksae/tests/test_lista.py` comparing the module output to a manual reference implementation (vector and batch), and validated `return_all` history shapes.
- Created a runnable example script `ksae/examples/run_lista.py` that runs LISTA on toy data and prints intermediate states.
- Couldn’t run tests here due to missing `pytest` and likely heavy PyTorch install; provided commands to run locally.

### Quick start tomorrow
- Set up a venv and run tests:
```bash
cd /home/mila/l/lia/ksae
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```
- Run the example:
```bash
python -m ksae.examples.run_lista --input-dim 16 --code-dim 8 --steps 3 --batch 4
```

### Next steps (recommended)
- Add a minimal training loop to fit LISTA to synthetic CoD targets:
  - Implement a small CoD solver for targets or stub with ISTA for now.
  - Train `W_e, S, theta` via MSE-to-targets, verify convergence with tests.
- Optional: add gradient tests (finite-diff on small tensors) for `LISTA` to ensure backprop behaves as expected.

### Pointers
- Main encoder: `ksae/src/ksae/lista/encoder.py`
- Tests: `ksae/tests/test_lista.py`, `ksae/tests/test_shrinkage.py`
- Example: `ksae/examples/run_lista.py`