# Corner-channel bounds

Python/CVXPY code for numerical bounds on quantum-channel d-compatibility:
an ordinary Choi-matrix X/Y seesaw lower search and a symmetry-reduced,
polarization-inspired PPT outer SDP. The upper SDP is specialized to (n,d)=(3,2).

```bash
python -m pip install -r requirements.txt
python -m pip install mosek==11.0.7  # optional; requires a license
python p23_standalone.py --test
python p23_standalone.py --audit --model lower_model.npz
python p23_standalone.py --lower --model warm_start.npz --p 0.8854
python p23_standalone.py --upper
```

Use Python 3.12 or newer. Without `--model`, the lower search starts randomly;
add `--restarts 3`. MOSEK is preferred; CLARABEL/SCS are fallbacks.
The upper solve can be expensive. There are no joint linearized updates.

`warm_start.npz` is a valid model at p=0.885, derived from `lower_model.npz`
by adding input depolarizing noise to the parent; the recoveries are unchanged.
Locally, ordinary seesaw recovered p=0.8854 in one X/Y cycle (MOSEK,
maximum Choi residual 8.1e-10). This is warm-start recovery, not a fresh
discovery: the original witness came from the earlier hybrid search.

Numerical endpoints for (3,2): lower **0.8854** (24 parent outcomes),
simple upper **0.8938254**. These are not exact-arithmetic certificates;
random searches need not recover the saved model, and the outer relaxation
is not the full convergent polarization hierarchy.

## Acknowledgment

The code was rewritten by OpenAI Codex and checked by the human author.
