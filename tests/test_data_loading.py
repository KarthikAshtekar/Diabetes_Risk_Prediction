from __future__ import annotations

from sklearn.model_selection import train_test_split

from brfss_diabetes.config import TARGET
from brfss_diabetes.data_loading import load_brfss_data


def test_data_loads_and_target_is_valid() -> None:
    frame = load_brfss_data()
    assert frame.shape == (253680, 22)
    assert TARGET in frame
    assert set(frame[TARGET].unique()) == {0, 1, 2}


def test_split_is_stratified() -> None:
    frame = load_brfss_data()
    _, _, y_train, y_test = train_test_split(
        frame.drop(columns=TARGET),
        frame[TARGET],
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=42,
    )
    full_share = frame[TARGET].value_counts(normalize=True).sort_index()
    train_share = y_train.value_counts(normalize=True).sort_index()
    test_share = y_test.value_counts(normalize=True).sort_index()
    assert (full_share - train_share).abs().max() < 0.001
    assert (full_share - test_share).abs().max() < 0.001


def test_loader_allows_one_missing_expected_predictor(tmp_path) -> None:
    frame = load_brfss_data().head(20).drop(columns="Fruits")
    path = tmp_path / "brfss_missing_optional_predictor.csv"
    frame.to_csv(path, index=False)

    loaded = load_brfss_data(path)

    assert TARGET in loaded
    assert "Fruits" not in loaded
    assert loaded.shape == (20, 21)
