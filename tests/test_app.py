from argparse import Namespace

import numpy as np

import src.app as app


def test_parse_args_reads_command_line_options():
    args = app.parse_args(["--words", "12", "--steps", "30", "--no-viz"])

    assert args.words == 12
    assert args.steps == 30
    assert args.no_viz is True


def test_main_headless_mode_runs_and_reports_completion(monkeypatch, capsys):
    fake_simulation = type(
        "FakeSimulation",
        (),
        {
            "step": lambda self: None,
            "vel": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "step_count": 2,
        },
    )()

    monkeypatch.setattr(
        app,
        "build_simulation",
        lambda words: (
            fake_simulation,
            {},
            [],
            np.empty((0, 0), dtype=float),
            [],
            [],
        ),
    )

    app.main(Namespace(words=2, steps=2, no_viz=True))

    captured = capsys.readouterr().out
    assert "Headless run complete" in captured
    assert "mean velocity" in captured
