import numpy as np
import pytest
from kkt_agent.qp_generator import QP
from kkt_agent.kkt_verifier import verify
from kkt_agent.residual_audit import compare_residuals


def cancellation_case():
    qp = QP(np.eye(2), np.zeros(2), np.array([[1., 0.], [1., 0.], [0., 1.]]), np.ones(3))
    x, lam = np.zeros(2), np.array([1e6, -1e6, 0.])
    return qp, x, lam, verify(qp, x, lam)


def test_cancellation_roundoff_is_logged_without_changing_certificate():
    qp, x, lam, cert = cancellation_case()
    record = cert.to_dict()
    record["raw_stationarity"] = 1e-10
    differences = compare_residuals(cert, record, qp, x, lam)
    assert [item["metric"] for item in differences] == ["raw_stationarity"]
    assert not cert.success  # Huge negative multiplier must still fail KKT.


def test_material_raw_corruption_is_rejected():
    qp, x, lam, cert = cancellation_case()
    record = cert.to_dict()
    record["raw_stationarity"] = 1e-2
    with pytest.raises(AssertionError, match="raw_stationarity"):
        compare_residuals(cert, record, qp, x, lam)


def test_normalized_tolerance_is_not_relaxed():
    qp, x, lam, cert = cancellation_case()
    record = cert.to_dict()
    record["primal"] = 1e-8
    with pytest.raises(AssertionError, match="primal"):
        compare_residuals(cert, record, qp, x, lam)
