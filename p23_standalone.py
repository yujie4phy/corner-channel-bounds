#!/usr/bin/env python3
"""Corner-channel bounds. Dependencies: numpy scipy cvxpy; optional mosek.
Run: --lower / --upper / --all / --audit --model witness.npz / --test.
Ordinary seesaw: --p 0.8854 --r 24; add --model witness.npz to warm-start.
Without --model, use random starts; --maximize-p selects exact-equality seesaw.
We choose r=24; the general Caratheodory cap n^4*binom(n,2) is 243 for n=3.
Limited r=36,48 trials found no improvement, not evidence of optimality.
All algorithm code is here; optional NPZ files contain numerical matrices only.
"""
import argparse
import itertools
import json
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path

license_path = Path(os.environ.get("MOSEKLM_LICENSE_FILE", str(Path.home()/".mosek/mosek.lic")))
license_path = license_path/"mosek.lic" if license_path.is_dir() else license_path
if license_path.is_file():
    os.environ["MOSEKLM_LICENSE_FILE"] = str(license_path)
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

DEFAULT_R = 24

# Unnormalized Choi matrices; zero-based edges; row-major tensor indices.
def edges(n):
    return list(itertools.combinations(range(n), 2))

def hermitian(M):
    return (M + M.conj().T)/2

def caratheodory_outcome_bound(n):
    return n**4 * len(edges(n))  # Sufficient, not minimal; original A=B=C^n.

def target_choi(n, edge, p, output_dim=None):
    P = np.diag(np.isin(np.arange(n), edge).astype(float))
    v = P.reshape(-1)
    J = p*np.outer(v,v) + np.kron(np.eye(n)-p*P,P/2)
    if output_dim is None or output_dim == n:
        return J
    if output_dim != 2:
        raise ValueError("Output dimension must be n or 2")
    ix = [a*n+b for a in range(n) for b in edge]
    return J[np.ix_(ix,ix)]

def apply_map(T, M, size):
    if isinstance(M,cp.Expression):
        return cp.reshape(T @ cp.vec(M,order="C"),(size,size),order="C")
    return (T @ np.asarray(M).reshape(-1)).reshape(size,size)

@lru_cache(maxsize=64)
def trace_map(dims, keep):
    if tuple(sorted(set(keep))) != keep or any(k < 0 or k >= len(dims) for k in keep):
        raise ValueError("keep must list retained factors in increasing order")
    D = int(np.prod(dims))
    digits = np.array(np.unravel_index(np.arange(D),dims))
    mask = np.ones((D,D),bool)
    for k in set(range(len(dims)))-set(keep):
        mask &= digits[k,:,None] == digits[k,None,:]
    i,j = np.nonzero(mask)
    out_dims = tuple(dims[k] for k in keep)
    m = int(np.prod(out_dims))
    labels = np.ravel_multi_index(digits[list(keep)],out_dims) if keep else np.zeros(D,int)
    return sp.coo_matrix((np.ones(len(i)),(labels[i]*m+labels[j],i*D+j)),
                         shape=(m*m,D*D)).tocsr(), m

def partial_trace(M, dims, keep):
    """Trace out every factor not listed in keep."""
    T,size = trace_map(tuple(dims),tuple(keep))
    return apply_map(T,M,size)

@lru_cache(maxsize=16)
def transpose_map(dims, factor):
    D,k = int(np.prod(dims)),len(dims)
    axes = list(range(2*k))
    axes[factor],axes[k+factor] = axes[k+factor],axes[factor]
    columns = np.arange(D*D).reshape(dims+dims).transpose(axes).reshape(-1)
    return sp.csr_matrix((np.ones(D*D),(np.arange(D*D),columns)),shape=(D*D,D*D))

def partial_transpose(M, dims, factor):
    return apply_map(transpose_map(tuple(dims),factor),M,int(np.prod(dims)))

@lru_cache(maxsize=16)
def link_indices(n, d, b):
    a,B,A,bb,c,C = np.indices((n,b,n,b,d,d)).reshape(6,-1)
    return (a*b+B)*(n*b)+A*b+bb, (a*d+c)*(n*d)+A*d+C, (c*b+B)*(d*b)+C*b+bb

def link_product(X, Y, n, d):
    b = Y.shape[0]//d
    return np.einsum("acij,cbjk->abik",X.reshape(n,d,n,d),Y.reshape(d,b,d,b)).reshape(n*b,n*b)

def link_product_fixed_Y(X, Y, n, d):
    b = Y.shape[0]//d
    out,ix,iy = link_indices(n,d,b)
    T = sp.coo_matrix((Y.reshape(-1)[iy],(out,ix)),shape=((n*b)**2,(n*d)**2)).tocsr()
    return apply_map(T,X,n*b)

def link_product_fixed_X(X, Y, n, d):
    b = Y.shape[0]//d
    out,ix,iy = link_indices(n,d,b)
    T = sp.coo_matrix((X.reshape(-1)[ix],(out,iy)),shape=((n*b)**2,(d*b)**2)).tocsr()
    return apply_map(T,Y,n*b)

def linearized_link_Z(Z, n, d, b=2):
    a,B,c = np.indices((n,b,d)).reshape(3,-1)
    V = sp.coo_matrix((np.ones(len(a)),(a*b+B,((a*d+c)*d+c)*b+B)),
                      shape=(n*b,n*d*d*b)).tocsr()
    return V @ Z @ V.T

# Solver selection and independent numerical checks.
def solve(problem, solver, verbose=False, lower=False):
    options = {"MOSEK": {} if lower else dict(eps=1e-8,mosek_params={"MSK_IPAR_NUM_THREADS":4}),
               "CLARABEL": dict(tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,max_iter=10000,
                                static_regularization_constant=1e-7),
               "SCS": dict(eps=1e-7 if lower else 1e-6,max_iters=100000 if lower else 20000)}
    try:
        problem.solve(solver=solver,verbose=verbose,**options[solver])
    except Exception as error:
        warnings.warn(f"{solver} failed: {error}")
        return False
    if problem.status != cp.OPTIMAL:
        warnings.warn(f"{solver}: {problem.status}; check residuals and tolerances")
    return problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE)

def choose_solver(requested=None):
    for solver in ([requested] if requested else ["MOSEK","CLARABEL","SCS"]):
        if solver in cp.installed_solvers():
            x = cp.Variable()
            if solve(cp.Problem(cp.Minimize(x),[x >= 1]),solver):
                return solver
    raise RuntimeError("No usable requested solver; install/license MOSEK, CLARABEL or SCS")

def audit(p, X, Y, n, d):
    b = next(iter(Y.values())).shape[0]//d
    residuals = [float(np.linalg.norm(target_choi(n,e,p,b)-sum(
        link_product(x,Y[e,l],n,d) for l,x in enumerate(X)))) for e in edges(n)]
    tp_X = float(np.linalg.norm(sum(partial_trace(x,(n,d),(0,)) for x in X)-np.eye(n)))
    tp_Y = float(max(np.linalg.norm(partial_trace(y,(d,b),(0,))-np.eye(d)) for y in Y.values()))
    matrices = [*X,*Y.values()]
    herm_error = float(max(np.linalg.norm(M-M.conj().T) for M in matrices))
    mineig = float(min(np.linalg.eigvalsh(hermitian(M)).min() for M in matrices))
    accepted = np.isfinite([p,*residuals,tp_X,tp_Y,herm_error,mineig]).all()
    accepted &= 0 <= p <= 1 and max(residuals) <= 2e-8 and max(tp_X,tp_Y,herm_error) <= 1e-7 and mineig >= -1e-7
    return dict(p=float(p),r=len(X),residuals=residuals,max_choi_residual=max(residuals),
                parent_trace_error=tp_X,recovery_trace_error=tp_Y,hermitian_error=herm_error,
                min_eigenvalue=mineig,accepted=bool(accepted))

def verify_model(p, X, Y, n, d):
    """Check raw Choi constraints and, independently, full-output Kraus composition."""
    def kraus(J,a,b):
        w,V = np.linalg.eigh(hermitian(J))
        return [np.sqrt(t)*v.reshape(a,b).T for t,v in zip(w,V.T) if t > 0]
    report = audit(p,X,Y,n,d)
    parents = [kraus(x,n,d) for x in X]
    b = next(iter(Y.values())).shape[0]//d
    residuals,tp_errors = [],[]
    for e in edges(n):
        embed = np.eye(n)[:,e] if b == 2 else np.eye(n)
        P = np.diag(np.isin(np.arange(n),e).astype(float))
        actual,target,total = np.zeros((n*n,n*n),complex),np.zeros((n*n,n*n)),np.zeros((n,n),complex)
        for l,N in enumerate(parents):
            for R in kraus(Y[e,l],d,b):
                for K in N:
                    L = embed @ R @ K
                    v = L.T.reshape(-1)
                    actual += np.outer(v,v.conj())
                    total += L.conj().T @ L
        for a,aa in itertools.product(range(n),repeat=2):
            E = np.zeros((n,n))
            E[a,aa] = 1
            target[a*n:(a+1)*n,aa*n:(aa+1)*n] = p*P @ E @ P + np.trace((np.eye(n)-p*P) @ E)*P/2
        residuals.append(float(np.linalg.norm(actual-target)))
        tp_errors.append(float(np.linalg.norm(total-np.eye(n))))
    report["kraus_residual"] = max(residuals)
    report["kraus_trace_error"] = max(tp_errors)
    report["accepted"] &= max(residuals) <= 2e-8 and max(tp_errors) <= 1e-7
    return report

# Recovery outputs are always compressed to B_e=C^2 during optimization.
def random_cptp(a, b, rng):
    rank = max(a,b)
    V,_ = np.linalg.qr(rng.normal(size=(rank*b,a))+1j*rng.normal(size=(rank*b,a)))
    return sum(np.outer(K.T.reshape(-1),K.T.reshape(-1).conj()) for K in V.reshape(rank,b,a))

def random_model(n, d, r, rng):
    J = random_cptp(n,r*d,rng)
    indices = [[a*r*d+l*d+c for a in range(n) for c in range(d)] for l in range(r)]
    X = [J[np.ix_(ix,ix)] for ix in indices]
    return X,{(e,l):random_cptp(d,2,rng) for e in edges(n) for l in range(r)}

def compress_recoveries(Y, n, d):
    if next(iter(Y.values())).shape[0] == 2*d:
        return {k:y.copy() for k,y in Y.items()}
    result = {}
    for (e,l),y in Y.items():
        ix = [c*n+b for c in range(d) for b in e]
        outside = np.zeros((d,d),complex)
        for b in set(range(n))-set(e):
            jx = [c*n+b for c in range(d)]
            outside += y[np.ix_(jx,jx)]
        result[e,l] = hermitian(y[np.ix_(ix,ix)] + np.kron(outside,np.eye(2)/2))
    return result

def load_model(path, n, d):
    with np.load(path,allow_pickle=False) as f:
        r,b = len(f["X"]),f["Y"].shape[-1]//d
        if ((int(f["n"]),int(f["d"])) != (n,d) or not np.array_equal(f["edges"],edges(n))
            or f["X"].shape != (r,n*d,n*d) or b not in (2,n)
            or f["Y"].shape != (len(edges(n)),r,d*b,d*b)):
            raise ValueError("Saved model dimensions or edges disagree")
        return float(f["p"]),list(f["X"]),{(e,l):f["Y"][j,l] for j,e in enumerate(edges(n)) for l in range(r)}

def grow_parent(X0, Y0, n, d, r, rng, perturb=0, method="split"):
    if r < len(X0) or perturb < 0 or method not in ("split","append"):
        raise ValueError("Need r>=saved outcomes, perturb>=0, growth=split/append")
    Y0 = compress_recoveries(Y0,n,d)
    counts = np.ones(len(X0),int)
    if method == "split":
        weights = np.maximum([np.trace(x).real for x in X0],0)
        for _ in range(r-len(X0)):
            counts[np.argmax(weights/counts)] += 1
    X,Y = [],{}
    for old,count in enumerate(counts):
        generators = {}
        for copy in range(count):
            l = len(X)
            X.append(X0[old]/count)
            for e in edges(n):
                y = Y0[e,old].copy()
                if perturb and (copy//2)*2+1 < count:
                    if copy % 2 == 0:
                        H = hermitian(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
                        generators[e] = H/np.linalg.norm(H)
                    w,V = np.linalg.eigh(generators[e])
                    U = (V*np.exp(1j*perturb*(-1)**copy*w)) @ V.conj().T
                    K = np.kron(np.eye(d),U)
                    y = hermitian(K @ y @ K.conj().T)
                Y[e,l] = y
    while len(X) < r:
        l = len(X)
        X.append(np.zeros_like(X0[0]))
        for e in edges(n):
            Y[e,l] = random_cptp(d,2,rng)
    return X,Y

# Lower bound: alternate X and Y SDPs; no joint linearization.
def seesaw_step(X0, Y0, n, d, p, update, solver):
    if update not in ("X","Y"):
        raise ValueError("update must be X or Y")
    def trace_output(M,a,b):
        return cp.bmat([[sum(M[i*b+k,j*b+k] for k in range(b)) for j in range(a)] for i in range(a)])
    X,Y,constraints = X0,Y0,[]
    if update == "X":
        X = [cp.Variable((n*d,n*d),hermitian=True) for _ in X0]
        constraints += [x >> 0 for x in X] + [sum(trace_output(x,n,d) for x in X) == np.eye(n)]
    else:
        Y = {k:cp.Variable((2*d,2*d),hermitian=True) for k in Y0}
        for y in Y.values():
            constraints += [y >> 0,trace_output(y,d,2) == np.eye(d)]
    p_var = cp.Variable(name="p") if p is None else p
    residuals = []
    for e in edges(n):
        link = sum(link_product_fixed_Y(X[l],Y0[e,l],n,d) if update == "X" else
                   link_product_fixed_X(X0[l],Y[e,l],n,d) for l in range(len(X)))
        J0,J1 = target_choi(n,e,0,2),target_choi(n,e,1,2)
        residuals.append(link-(J0+p_var*(J1-J0)))
    if p is None:
        constraints += [p_var >= 0,p_var <= 1] + [R == 0 for R in residuals]
        objective = cp.Maximize(p_var)
    else:
        objective = cp.Minimize(sum(cp.norm(cp.hstack([cp.real(R),cp.imag(R)]),"fro") for R in residuals))
    problem = cp.Problem(objective,constraints)
    if not solve(problem,solver,lower=True):
        return None
    value = lambda M: hermitian(M.value) if isinstance(M,cp.Expression) else M
    return [value(x) for x in X],{k:value(y) for k,y in Y.items()},float(p_var.value) if p is None else p,problem.status

def solve_seesaw(n=3, d=2, r=DEFAULT_R, p=0.8854, restarts=1, iterations=30,
                  seed=0, solver="MOSEK", model_path=None, perturb=0, growth="split"):
    rng,best,history,statuses = np.random.default_rng(seed),None,[],set()
    started = time.monotonic()
    for restart in range(restarts):
        if model_path is None:
            X,Y = random_model(n,d,r,rng)
        else:
            _,X,Y = load_model(model_path,n,d)
            X,Y = grow_parent(X,Y,n,d,r,rng,perturb,growth)
        previous = None
        for cycle in range(iterations):
            for update in ("X","Y"):
                result = seesaw_step(X,Y,n,d,p,update,solver)
                if result is None:
                    statuses.add("failed_"+update)
                    break
                X,Y,p_value,status = result
                statuses.add(status)
            if result is None:
                break
            report = audit(p_value,X,Y,n,d)
            score = (not report["accepted"],-p_value if p is None else sum(report["residuals"]))
            if best is None or score < best[0]:
                best = score,report,X,Y
            history.append(dict(restart=restart+1,cycle=cycle+1,**report))
            print(f"restart {restart+1}, cycle {cycle+1}: p={p_value:.12f}, residual={report['max_choi_residual']:.3e}",flush=True)
            if report["accepted"] and (p is not None or previous is not None and abs(p_value-previous) < 1e-8):
                break
            previous = p_value
    if best is None:
        return dict(accepted=False,statuses=sorted(statuses)),None
    _,report,X,Y = best
    report = verify_model(report["p"],X,Y,n,d)
    report.update(solver=solver,statuses=sorted(statuses),history=history,seconds=time.monotonic()-started)
    return report,(X,Y)

# Exact S3 block reduction for the (3,2) upper SDP.
def basis_action(index, perm):
    a,c,*bits = np.unravel_index(index,(3,2,2,2,2,2,2,2))
    E,moved = edges(3),[None]*3
    for k,e in enumerate(E):
        image = tuple(sorted(perm[v] for v in e))
        moved[E.index(image)] = bits[2*k],image.index(perm[e[bits[2*k+1]]])
    return np.ravel_multi_index((perm[a],c,*itertools.chain.from_iterable(moved)),(3,2,2,2,2,2,2,2))

def assemble_basis(columns):
    rows,cols,data = [],[],[]
    for j,(indices,values) in enumerate(columns):
        rows.extend(indices)
        cols.extend([j]*len(indices))
        data.extend(values)
    return sp.coo_matrix((data,(rows,cols)),shape=(384,len(columns))).tocsr()

@lru_cache(maxsize=1)
def symmetry_bases():
    permutations = list(itertools.permutations(range(3)))
    seen,columns = set(),[[],[],[],[]]
    for i in range(384):
        if i in seen:
            continue
        orbit = sorted({basis_action(i,g) for g in permutations})
        seen.update(orbit)
        m,pos = len(orbit),{x:j for j,x in enumerate(orbit)}
        t = np.ones(m)/np.sqrt(m)
        columns[0].append((orbit,t))
        P = np.eye(m)-np.outer(t,t)
        if m == 6:
            s = np.zeros(m)
            for g in permutations:
                s[pos[basis_action(orbit[0],g)]] = (-1)**sum(g[a]>g[b] for a in range(3) for b in range(a+1,3))/np.sqrt(6)
            columns[1].append((orbit,s))
            P -= np.outer(s,s)
        def representation(g):
            return np.eye(m)[:,[pos[basis_action(x,g)] for x in orbit]]
        reflection,rotation = representation((0,2,1)),representation((1,2,0))
        w,V = np.linalg.eigh(hermitian(P @ (np.eye(m)+reflection)/2))
        plus = V[:,w > 0.5]
        minus = 2/np.sqrt(3)*(rotation+0.5*np.eye(m)) @ plus
        for j in range(plus.shape[1]):
            columns[2].append((orbit,plus[:,j]))
            columns[3].append((orbit,minus[:,j]))
    return tuple(assemble_basis(c) for c in columns)

def edge_stabilizer_bases():
    seen,even,odd = set(),[],[]
    for i in range(384):
        if i in seen:
            continue
        j = basis_action(i,(1,0,2))
        seen.update((i,j))
        even.append(([i],[1.0]) if i == j else ([i,j],[1/np.sqrt(2),1/np.sqrt(2)]))
        if i != j:
            odd.append(([i,j],[1/np.sqrt(2),-1/np.sqrt(2)]))
    return assemble_basis(even),assemble_basis(odd)

def build_outer_relaxation():
    """W on (A,C0)|(C12,B12)|(C13,B13)|(C23,B23), dimensions (6,4,4,4).
    Exact models: W=sum_l (X_l/3) tensor (Y12_l/2) tensor (Y13_l/2) tensor (Y23_l/2).
    Keep affine marginal constraints; replace separability by whole-factor PPT.
    This finite outer relaxation is not the complete convergent hierarchy.
    """
    dims = (6,4,4,4)
    Qt,Qs,Qp,Qm = symmetry_bases()
    blocks = [cp.Variable((Q.shape[1],Q.shape[1]),symmetric=True) for Q in (Qt,Qs,Qp)]
    embeddings = [sp.kron(Qt,Qt,format="csr"),sp.kron(Qs,Qs,format="csr"),
                  sp.kron(Qp,Qp,format="csr")+sp.kron(Qm,Qm,format="csr")]
    def mapped(T,size):
        return cp.reshape(sum((T @ E) @ cp.vec(M,order="C") for E,M in zip(embeddings,blocks)),
                          (size,size),order="C")
    S,Z_e,U_ef = [mapped(*trace_map(dims,k)) for k in ((0,),(0,1),(0,1,2))]
    p = cp.Variable(name="p")
    J0,J1 = [target_choi(3,(0,1),t,2) for t in (0,1)]
    constraints = [p >= 0,p <= 1,
        sum(k*cp.trace(M) for k,M in zip((1,1,2),blocks)) == 1,
        partial_trace(S,(3,2),(0,)) == np.eye(3)/3,
        partial_trace(Z_e,(6,2,2),(0,1)) == cp.kron(S,np.eye(2)/2),
        linearized_link_Z(Z_e,3,2) == (J0+p*(J1-J0))/6,
        partial_trace(U_ef,(6,4,2,2),(0,1,2)) == cp.kron(Z_e,np.eye(2)/2)]
    psd = list(blocks)
    for factor,bases in ((0,(Qt,Qs,Qp)),(1,edge_stabilizer_bases())):
        for Q in bases:
            T = sp.kron(Q.T,Q.T,format="csr") @ transpose_map(dims,factor)
            psd.append(mapped(T,Q.shape[1]))
    constraints += [M >> 0 for M in psd]
    # Other edges follow by S3; PPT of U_ef follows by tracing global PPT.
    return cp.Problem(cp.Maximize(p),constraints),blocks,psd

def solve_outer_relaxation(n=3, d=2, solver="MOSEK", verbose=False):
    if (n,d) != (3,2):
        raise ValueError("The symmetry-reduced upper SDP is specialized to (3,2)")
    problem,_,psd = build_outer_relaxation()
    print(f"Upper SDP: {solver}, blocks 64,64,128; eight PSD cones",flush=True)
    if not solve(problem,solver,verbose):
        return dict(accepted=False,status=problem.status)
    violation = max(float(np.max(c.violation())) for c in problem.constraints)
    mineig = min(float(np.linalg.eigvalsh(hermitian(M.value)).min()) for M in psd)
    accepted = problem.status == cp.OPTIMAL and np.isfinite([problem.value,violation,mineig]).all()
    return dict(p=float(problem.value),status=problem.status,solver=solver,
                accepted=bool(accepted and violation <= 1e-7 and mineig >= -1e-7),
                max_constraint_violation=violation,min_eigenvalue=mineig,
                seconds=problem.solver_stats.solve_time)

def self_test():
    rng = np.random.default_rng(7)
    assert caratheodory_outcome_bound(3) == 243
    for n,d in ((3,2),(4,2),(4,3)):
        X,Y = random_cptp(n,d,rng),random_cptp(d,2,rng)
        expected = link_product(X,Y,n,d)
        for actual in (link_product_fixed_X(X,cp.Constant(Y),n,d).value,
                       link_product_fixed_Y(cp.Constant(X),Y,n,d).value,
                       linearized_link_Z(np.kron(X,Y),n,d)):
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(partial_trace(expected,(n,2),(0,)),np.eye(n),atol=1e-12)
        np.testing.assert_allclose(partial_transpose(np.kron(X,Y),(n*d,2*d),0),np.kron(X.T,Y))
        for e in edges(n):
            for p in (0,0.5,1):
                J = target_choi(n,e,p)
                np.testing.assert_allclose(partial_trace(J,(n,n),(0,)),np.eye(n))
                assert np.linalg.eigvalsh(J).min() > -1e-12
    X = [np.kron(np.eye(3),np.eye(2)/2)]
    Y = {(e,0):np.eye(4)/2 for e in edges(3)}
    assert verify_model(0,X,Y,3,2)["accepted"]
    for method in ("split","append"):
        Xg,Yg = grow_parent(X,Y,3,2,24,rng,method=method)
        assert audit(0,Xg,Yg,3,2)["accepted"]
    bases = symmetry_bases()
    Q = sp.hstack(bases).toarray()
    np.testing.assert_allclose(Q.T @ Q,np.eye(384),atol=1e-12)
    problem,blocks,_ = build_outer_relaxation()
    for M in blocks:
        M.value = np.eye(M.shape[0])/384
    problem.objective.args[0].value = 0
    assert problem.is_dcp() and max(float(np.max(c.violation())) for c in problem.constraints) < 1e-12
    print("PASS: Choi/link/trace/PT, Kraus audit, outcome growth, symmetry and lifted p=0 model")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for name in ("lower","upper","all","audit","test"):
        mode.add_argument("--"+name,action="store_true")
    for name,default in (("n",3),("d",2),("r",DEFAULT_R),("restarts",1),("iterations",30),("seed",0)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--p",type=float,default=0.8854)
    parser.add_argument("--perturb",type=float,default=0)
    parser.add_argument("--growth",choices=("split","append"),default="split")
    parser.add_argument("--maximize-p",action="store_true")
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--solver",choices=("MOSEK","CLARABEL","SCS"))
    for name in ("model","save","report"):
        parser.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    if args.test:
        self_test()
        return
    if not (args.n >= 2 and 1 <= args.d <= args.n and min(args.r,args.restarts,args.iterations) >= 1
            and 0 <= args.p <= 1 and args.perturb >= 0):
        parser.error("Invalid dimensions, outcome/iteration counts, p or perturbation")
    if args.audit:
        if args.model is None:
            parser.error("--audit requires --model witness.npz")
        report = {"audit":verify_model(*load_model(args.model,args.n,args.d),args.n,args.d)}
    else:
        if (args.upper or args.all) and (args.n,args.d) != (3,2):
            parser.error("The upper SDP currently supports only (n,d)=(3,2)")
        solver = choose_solver(args.solver)
        report = {"q_d_n":(args.d*args.n-1)/(args.n**2-1)}
        if not args.upper:
            report["caratheodory_outcome_bound"] = caratheodory_outcome_bound(args.n)
            print(f"Parent outcomes: r={args.r}; general sufficient cap {report['caratheodory_outcome_bound']}",flush=True)
            report["lower"],model = solve_seesaw(args.n,args.d,args.r,None if args.maximize_p else args.p,
                args.restarts,args.iterations,args.seed,solver,args.model,args.perturb,args.growth)
            if args.save and model is not None:
                X,Y = model
                np.savez_compressed(args.save,n=args.n,d=args.d,r=len(X),p=report["lower"]["p"],
                    output_dim=2,accepted=report["lower"]["accepted"],edges=edges(args.n),X=X,
                    Y=np.array([[Y[e,l] for l in range(len(X))] for e in edges(args.n)]))
        if args.upper or args.all:
            report["upper"] = solve_outer_relaxation(args.n,args.d,solver,args.verbose)
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    if not all(report[k]["accepted"] for k in ("lower","upper","audit") if k in report):
        raise SystemExit("Audit failed: do not interpret this run as a bound")
    print("Numerical estimates only, not exact primal/dual certificates.")

if __name__ == "__main__":
    main()
