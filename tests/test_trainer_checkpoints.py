"""Пины политики чекпоинтов трейнера: на диск идут только ``last.pt`` и ``best.pt``.

Пер-эпохные копии (``epoch=NN-val_loss=X.pt``) писались при каждом улучшении val и
на длинных ранах давали десятки гигабайт (470 МБ × число улучшений). Кривые обучения
живут в Comet, поэтому копии не нужны — тесты фиксируют, что они не возвращаются.
"""

import trainer


def test_checkpoint_paths_without_improvement_writes_last_only() -> None:
    assert trainer._checkpoint_paths("/ckpt", improved=False) == ["/ckpt/last.pt"]


def test_checkpoint_paths_on_improvement_adds_best() -> None:
    assert trainer._checkpoint_paths("/ckpt", improved=True) == ["/ckpt/last.pt", "/ckpt/best.pt"]


def test_checkpoint_paths_never_emit_per_epoch_tags() -> None:
    for improved in (False, True):
        paths = trainer._checkpoint_paths("/ckpt", improved=improved)
        assert not any("epoch=" in path for path in paths)
