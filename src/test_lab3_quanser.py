"""Быстрые автономные проверки; синтетика применяется только внутри тестов."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.signal import chirp
import lab3_quanser_core as c


def test_transfer_static_limit():
    response = c.response(np.array([0.0]), c.NOMINAL)
    assert np.allclose(response[0], [1.0, 1.0], atol=1e-9)
    empty = c.response(np.array([3.0]), c.NOMINAL, np.zeros(4))
    assert np.isfinite(empty).all()
    assert np.allclose(empty, 0)


def test_two_dimensional_ode_and_spectrum():
    t = np.linspace(0, 14, 1401)
    r = 0.004*chirp(t, f0=.5, f1=7, t1=14, method='linear')
    empty = c.Experiment(t, np.zeros_like(t), np.zeros_like(t), r, 'synthetic_test', '')
    states = c.simulate(empty, c.NOMINAL, np.ones(4))
    assert states.shape == (len(t), 4)
    exp = c.Experiment(t, states[:, 0], states[:, 1], r, 'synthetic_test', '')
    spec = c.make_spectrum(exp)
    assert spec.H.shape[1] == 2
    tr, va = c.split_spectrum(spec)
    theta = c.fit_graph(tr, np.ones(4))
    assert np.all(np.isfinite(theta))
    assert c.relative_error(va, theta) < 1.0
    return exp


def test_xlsx_parser_and_intervention():
    exp = test_two_dimensional_ode_and_spectrum()
    with tempfile.TemporaryDirectory() as tmp:
        names = ("Body Displacement Data.xlsx", "Tire Position Data.xlsx", "Vertical Displacement Error.xlsx")
        files = {}
        for name, y in zip(names, [exp.x_s, exp.x_u, exp.road-exp.x_s]):
            raw = pd.DataFrame({0: ["Variable frequency", "Time"]+exp.t.tolist(),
                                1: [np.nan, "Passive"]+y.tolist(),
                                2: [np.nan, "PID"]+[0.0]*len(y)})
            path = Path(tmp)/name
            raw.to_excel(path, header=False, index=False)
            files[name] = path
        joined = c.make_experiment(files, n_points=800)
        assert len(joined.t)==800
        assert np.corrcoef(joined.road, np.interp(joined.t, exp.t, exp.road))[0,1]>0.999
        observed = c.make_spectrum(joined)
        intervened = c.make_spectrum(c.Experiment(joined.t, joined.x_s, np.zeros_like(joined.x_u), joined.road,"zero",joined.info))
        p = c.smc(observed, n_particles=28, n_temper=5, mh_steps=1)
        q = c.smc(intervened, n_particles=28, n_temper=5, mh_steps=1)
        ks = c.compare_posteriors(p, q)
        tr, va = c.split_spectrum(observed)
        ranks = c.score_candidates(tr, va, ks)
        assert len(ranks)==16
        assert ranks.J.notna().all()
        g = c.dynamics_graph(c.mask_to_array(ranks.iloc[0]["mask"]))
        import networkx as nx
        assert nx.is_directed_acyclic_graph(g)
        flow, losses, samples = c.learn_graph_flow(ranks, epochs=8, n_samples=25)
        assert len(samples)==3
        assert len(losses)==8
        assert len(c.graph_sample_summary(samples,ranks))==3




def test_fetch_quanser_cache_and_403():
    """HTTP 403 не маскируется, локальные оригиналы читаются без сети."""
    from unittest.mock import patch
    import hashlib
    import requests

    with tempfile.TemporaryDirectory() as tmp:
        names = tuple(c.FILE_IDS)
        checksums = {}
        for name in names:
            path = Path(tmp) / name
            pd.DataFrame({"time": range(2000), "value": np.sin(np.arange(2000))}).to_excel(path, index=False)
            checksums[name] = hashlib.md5(path.read_bytes()).hexdigest()
        # Для теста подменяем ожидаемые суммы на суммы ТЕСТОВЫХ файлов;
        # в реальном запуске проверяются опубликованные Harvard MD5.
        with patch.dict(c.FILE_MD5, checksums):
            with patch.object(requests.Session, 'get', side_effect=AssertionError('Сеть не нужна')):
                files = c.fetch_quanser(tmp)
                assert len(files) == 3

            (Path(tmp) / names[0]).unlink()
            response = requests.Response()
            from unittest.mock import Mock
            response.raw = Mock()
            response.status_code = 403
            response.url = f"{c.API}/access/datafile/{c.FILE_IDS[names[0]]}"
            with patch.object(requests.Session, 'get', return_value=response):
                try:
                    c.fetch_quanser(tmp)
                    raise AssertionError('Ожидалась диагностическая ошибка 403')
                except RuntimeError as error:
                    msg = str(error)
                    assert '403' in msg
                    assert str(Path(tmp)) in msg
                    assert '12133334' in msg


if __name__ == "__main__":
    test_transfer_static_limit()
    test_two_dimensional_ode_and_spectrum()
    test_xlsx_parser_and_intervention()
    test_fetch_quanser_cache_and_403()
    print("ALL TESTS PASSED")


def test_regularization_and_computation_graph():
    """Новые части: выбор разных масок, арифметическая корректность графа, рендер."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import networkx as nx

    sample = pd.DataFrame([
        {"mask": "1111", "test_nrmse": 0.1, "edges": 4, "intervention_penalty": 0.0},
        {"mask": "0000", "test_nrmse": 1.0, "edges": 0, "intervention_penalty": 1.0},
    ])
    assert c.rescore_graphs(sample, 0.0, 0.5).iloc[0]["mask"] == "1111"
    assert c.rescore_graphs(sample, 2.0, 0.5).iloc[0]["mask"] == "0000"
    sweep = c.regularization_sweep(sample, [0, 2], [0.5])
    assert list(sweep["best_mask"]) == ["1111", "0000"]
    full = c.computation_graph(c.mask_to_array("1111"))
    empty = c.computation_graph(c.mask_to_array("0000"))
    assert nx.is_directed_acyclic_graph(full)
    assert full.has_edge("force_ks", "a_s")
    assert full.has_edge("force_ct", "a_u")
    assert not empty.has_node("force_ct")
    fig, ax = c.draw_computation_graph(c.mask_to_array("1010"))
    assert fig is ax.figure
    plt.close(fig)


if __name__ == "__main__":
    test_regularization_and_computation_graph()
    print("EXTENDED TESTS PASSED")
