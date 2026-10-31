import logging
from multiprocessing.pool import Pool

from dftio.__main__ import report_parse_results, wapper
from dftio.io.parse import Parser


class _TestParser(Parser):
    def __init__(self, root, failed_names=()):
        super().__init__(root=root, prefix="")
        self.loaded = self._load_valid_inputs(
            lambda idx: self._load(idx, failed_names)
        )

    def _load(self, idx, failed_names):
        if self.raw_datas[idx].split("/")[-1] in failed_names:
            raise ValueError
        return self.raw_datas[idx]

    def get_structure(self, idx):
        raise NotImplementedError

    def get_eigenvalue(self, idx, band_index_min):
        raise NotImplementedError

    def get_blocks(self, idx, hamiltonian=False, overlap=False, density_matrix=False):
        raise NotImplementedError


class _WriteParser:
    failed_inputs = ["bad_load.0"]

    def __getitem__(self, idx):
        return ["/input/good.0", "/input/bad_write.0"][idx]

    def write(self, idx, **kwargs):
        if idx == 1:
            raise RuntimeError


def test_eager_loading_excludes_failed_inputs():
    parser = _TestParser(
        root=["/input/good.0", "/input/bad.0", "/input/good.1"],
        failed_names={"bad.0"},
    )

    assert parser.raw_datas == ["/input/good.0", "/input/good.1"]
    assert parser.loaded == parser.raw_datas
    assert parser.failed_inputs == ["bad.0"]


def test_write_errors_are_skipped_and_reported_without_reason(caplog):
    parser = _WriteParser()
    worker = wapper({}, parser=parser)
    results = [worker(0), worker(1)]

    with caplog.at_level(logging.INFO):
        report_parse_results(parser, results)

    assert results == [(True, "good.0"), (False, "bad_write.0")]
    assert "Failed input: bad_load.0" in caplog.text
    assert "Failed input: bad_write.0" in caplog.text
    assert "RuntimeError" not in caplog.text
    assert "Parsing summary: total=3, succeeded=1, failed=2" in caplog.text


def test_write_errors_are_skipped_with_multiple_workers():
    with Pool(2) as pool:
        results = list(pool.imap(wapper({}, parser=_WriteParser()), range(2)))

    assert results == [(True, "good.0"), (False, "bad_write.0")]
