"""Roundoff allowances for comparing saved raw residuals across BLAS kernels.

These bounds do not enter the KKT verifier or change its success tolerance.
For float64 dot products, gamma_k = k*eps/(1-k*eps); 4*gamma_k covers
both evaluations plus additions, products and the reduction used in the bound.
Normalized residuals retain the original comparison tolerance.
"""
import numpy as np


def raw_roundoff_bounds(qp, x, lam):
    x, lam = np.asarray(x, dtype=float), np.asarray(lam, dtype=float)
    operations = qp.n + qp.m + 4
    epsilon = np.finfo(np.float64).eps
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    slack_scale = np.abs(qp.A) @ np.abs(x) + np.abs(qp.b)
    stationarity_scale = np.abs(qp.Q) @ np.abs(x) + np.abs(qp.c) + np.abs(qp.A.T) @ np.abs(lam)
    return {
        "raw_stationarity": float(4 * gamma * np.max(stationarity_scale)),
        "raw_primal": float(4 * gamma * np.max(slack_scale)),
        "raw_dual": 0.0,
        "raw_complementarity": float(4 * gamma * np.max(np.abs(lam) * slack_scale)),
    }


def compare_residuals(certificate, recorded, qp, x, lam):
    """Raise on material differences; return raw differences explained by roundoff."""
    bounds = raw_roundoff_bounds(qp, x, lam)
    reconciled = []
    for metric in ("stationarity", "primal", "dual", "complementarity",
                   "raw_stationarity", "raw_primal", "raw_dual", "raw_complementarity"):
        actual, saved = getattr(certificate, metric), recorded[metric]
        if np.isclose(actual, saved, rtol=1e-7, atol=1e-12):
            continue
        allowed = bounds.get(metric, 0.0)
        if not np.isclose(actual, saved, rtol=1e-7, atol=1e-12 + allowed):
            raise AssertionError(f"{metric} mismatch: recomputed={actual}, saved={saved}, roundoff_bound={allowed}")
        reconciled.append({"metric": metric, "absolute_difference": float(abs(actual-saved)),
                           "roundoff_bound": allowed})
    return reconciled
