"""P/R/F against resolution and number of views: values read at one threshold, declared cells only."""

from eval3d.figures.supplementary import curves as c

M = {"thresholds": [0.5, 1.0], "precision": [0.5, 0.9], "recall": [0.25, 0.8], "fscore": [0.3333, 0.847]}
DECLARED = {("a", "d2", None), ("a", "d4", None), ("a", "d4", 84), ("a", "d4", 5), ("b", "d4", None), ("b", "d4", 84)}


def test_prf_at_threshold_in_percent_and_failure():
    assert c.prf_at(M, 1.0) == (90.0, 80.0, 84.7)
    assert c.prf_at(None, 1.0) is None


def test_series_keeps_declared_cells_and_marks_failures():
    def declared(m, t, v):
        return (m, t, v) in DECLARED

    got = c.series(["a", "b"], declared, lambda m, t, v: None if (m, v) == ("a", 5) else M, ["d1", "d2", "d4"], [5, 84], "d4", 1.0)
    assert got["a"]["res"] == [(1, (90.0, 80.0, 84.7)), (2, (90.0, 80.0, 84.7))]
    assert got["a"]["views"] == [(5, None), (84, (90.0, 80.0, 84.7))]
    assert got["b"] == {"res": [(2, (90.0, 80.0, 84.7))], "views": []}      # no view subsets: no views curve
