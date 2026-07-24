from __future__ import annotations


def test_runtime_entrypoint_dependencies_import() -> None:
    import requests
    import tqdm

    from nhanes_feasibility.download_nhanes import download_nhanes_files

    assert requests.__version__
    assert tqdm.__version__
    assert callable(download_nhanes_files)
