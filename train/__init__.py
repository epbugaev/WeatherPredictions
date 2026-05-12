"""Specialised training entry points kept for historical experiments.

Every script in this package hardcodes its model / dataset choices and
defers the actual loop to ``train._common.run_legacy_training`` (which in
turn builds a ``trainer.Trainer``). The shared helper eliminates per-script
duplication while still keeping each ``train_<Model>.py`` self-contained.
"""
