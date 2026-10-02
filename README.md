# Corner-channel bounds

Python/CVXPY code for numerical bounds on quantum-channel d-compatibility:
an enhanced Choi-matrix seesaw lower search and a symmetry-reduced,
polarization-inspired PPT outer SDP. The upper SDP is specialized to (n,d)=(3,2).

```bash
python -m pip install -r requirements.txt
python -m pip install mosek==11.0.7  # optional; requires a license
python p23_standalone.py --test
python p23_standalone.py --audit --model lower_model.npz
python p23_standalone.py --lower --model lower_model.npz
python p23_standalone.py --upper
```

Use Python 3.12 or newer. Without `--model`, the lower search starts randomly;
add `--restarts 3`. Loading the supplied model refines an existing witness,
not a fresh random rediscovery. MOSEK is preferred; CLARABEL/SCS are fallbacks.
The upper solve can be expensive.

Numerical endpoints for (3,2): lower **0.8854** (24 parent outcomes),
simple upper **0.8938254**. These are not exact-arithmetic certificates;
random searches need not recover the saved model, and the outer relaxation
is not the full convergent polarization hierarchy.

## Acknowledgment

The code was rewritten by OpenAI Codex and checked by the human author.
