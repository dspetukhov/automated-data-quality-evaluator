"""Baseline regression tests for plot.py — captures CURRENT behavior as-is."""

from datetime import date

import polars as pl
import pytest
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

from plot import adjust_chart, create_chart, highlight_outliers, make_charts
from utility import TIME_INTERVAL_COL

# ---------------------------------------------------------------------------
# create_chart
# ---------------------------------------------------------------------------


class TestCreateChart:
    def test_n_cols_always_two(self):
        fig, n_cols, n_rows = create_chart(4, {}, titles=["a", "b", "c", "d"])
        assert n_cols == 2
        assert n_rows == 2

    def test_odd_subplot_count_rounds_rows_up(self):
        _, n_cols, n_rows = create_chart(3, {}, titles=["a", "b", "c"])
        assert n_cols == 2
        assert n_rows == 2

    def test_single_subplot_still_gets_one_row(self):
        _, n_cols, n_rows = create_chart(1, {}, titles=["a"])
        assert n_cols == 2
        assert n_rows == 1

    def test_titles_split_on_double_underscore_delimiter(self):
        titles = [
            " __ col_a __Number of unique values",
            " __ col_b __Proportion of missing values",
        ]
        fig, _, _ = create_chart(2, {}, titles=titles)
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert annotation_texts == [
            "Number of unique values",
            "Proportion of missing values",
        ]

    def test_title_without_delimiter_is_kept_as_is(self):
        fig, _, _ = create_chart(2, {}, titles=["plain title", "other"])
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert annotation_texts == ["plain title", "other"]

    def test_custom_spacing_config_is_applied(self):
        fig, _, _ = create_chart(
            2, {"horizontal_spacing": 0.3, "vertical_spacing": 0.4}, titles=["a", "b"]
        )
        # make_subplots does not expose spacing directly on the figure, so we
        # only assert construction succeeds with custom config (smoke test).
        assert isinstance(fig, Figure)

    def test_default_spacing_used_when_missing(self):
        # Should not raise even though config is empty.
        fig, _, _ = create_chart(2, {}, titles=["a", "b"])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# highlight_outliers
# ---------------------------------------------------------------------------


class TestHighlightOutliers:
    def _fig(self):
        return make_subplots(rows=2, cols=2)

    def test_no_shapes_added_when_lower_bound_is_none(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 2)])
        data = pl.Series([1, 2])
        result = highlight_outliers(fig, 0, x, data, (None, 5.0), 2, {})
        assert result.layout.shapes == ()

    def test_no_shapes_added_when_upper_bound_is_none(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 2)])
        data = pl.Series([1, 2])
        result = highlight_outliers(fig, 0, x, data, (1.0, None), 2, {})
        assert result.layout.shapes == ()

    def test_no_shapes_added_when_both_bounds_none(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 2)])
        data = pl.Series([1, 2])
        result = highlight_outliers(fig, 0, x, data, (None, None), 2, {})
        assert result.layout.shapes == ()

    def test_two_shapes_added_when_both_bounds_present(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 5)])
        data = pl.Series([1, 2, 3, 10])
        result = highlight_outliers(fig, 0, x, data, (2.0, 8.0), 2, {})
        assert len(result.layout.shapes) == 2

    def test_shape_coordinates_span_below_and_above_bounds(self):
        # shape[0] covers [data.min(), lower_bound]
        # shape[1] covers [upper_bound, data.max()]
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 5)])
        data = pl.Series([1, 2, 3, 10])
        result = highlight_outliers(fig, 0, x, data, (2.0, 8.0), 2, {})
        shapes = result.layout.shapes
        assert shapes[0].y0 == 1  # data.min()
        assert shapes[0].y1 == 2.0  # lower_bound
        assert shapes[1].y0 == 8.0  # upper_bound
        assert shapes[1].y1 == 10  # data.max()

    def test_shape_x_range_spans_full_x_series(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 3)])
        data = pl.Series([1, 2, 3])
        result = highlight_outliers(fig, 0, x, data, (0.5, 2.5), 2, {})
        shapes = result.layout.shapes
        assert shapes[0].x0 == min(x)
        assert shapes[0].x1 == max(x)

    def test_null_in_x_series_does_not_raise(self):
        # FIXED: x0/x1 used to be computed with Python's builtin min()/max(),
        # which raise TypeError when x contains a null (e.g. a null-keyed
        # group produced by grouping a date column with missing values).
        # Now uses Series.min()/max(), which skip nulls like data.min() above.
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), None, date(2024, 1, 5)])
        data = pl.Series([1, 2, 3])
        result = highlight_outliers(fig, 0, x, data, (0.5, 2.5), 2, {})
        shapes = result.layout.shapes
        assert shapes[0].x0 == date(2024, 1, 1)
        assert shapes[0].x1 == date(2024, 1, 5)

    def test_row_col_placement_derived_from_subplot_index(self):
        # subplot index 3 with n_cols=2 -> row 2, col 2 (1-indexed).
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 2)])
        data = pl.Series([1, 2])
        result = highlight_outliers(fig, 3, x, data, (0.0, 1.0), 2, {})
        shape = result.layout.shapes[0]
        assert shape.xref == "x4"
        assert shape.yref == "y4"

    def test_style_config_kwargs_forwarded_to_shape(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1), date(2024, 1, 2)])
        data = pl.Series([1, 2])
        result = highlight_outliers(
            fig, 0, x, data, (0.0, 1.0), 2, {"fillcolor": "red", "opacity": 0.3}
        )
        shape = result.layout.shapes[0]
        assert shape.fillcolor == "red"
        assert shape.opacity == 0.3

    def test_returns_same_figure_instance(self):
        fig = self._fig()
        x = pl.Series([date(2024, 1, 1)])
        data = pl.Series([1])
        result = highlight_outliers(fig, 0, x, data, (None, None), 2, {})
        assert result is fig


# ---------------------------------------------------------------------------
# adjust_chart
# ---------------------------------------------------------------------------


class TestAdjustChart:
    def _fig(self, titles=("A", "B")):
        return make_subplots(rows=1, cols=2, subplot_titles=list(titles))

    def test_default_layout_dimensions(self):
        fig = self._fig()
        result = adjust_chart(fig, n_cols=2, n_rows=1, config={})
        assert result.layout.width == 1400  # 700 * 2
        assert result.layout.height == 240  # 240 * 1

    def test_default_layout_scales_with_grid_size(self):
        fig = make_subplots(rows=2, cols=2)
        result = adjust_chart(fig, n_cols=2, n_rows=2, config={})
        assert result.layout.width == 1400
        assert result.layout.height == 480

    def test_custom_layout_overrides_defaults(self):
        fig = self._fig()
        result = adjust_chart(
            fig,
            2,
            1,
            {"layout": {"width": 100, "height": 50, "template": "plotly_dark"}},
        )
        assert result.layout.width == 200  # still multiplied by n_cols
        assert result.layout.height == 50  # still multiplied by n_rows (=1)
        assert result.layout.template.layout.annotationdefaults is not None

    def test_default_tickformat_applied_to_xaxes(self):
        fig = self._fig()
        result = adjust_chart(fig, 2, 1, {})
        assert result.layout.xaxis.tickformat == "%Y-%m-%d"
        assert result.layout.xaxis2.tickformat == "%Y-%m-%d"

    def test_custom_tickformat_applied(self):
        fig = self._fig()
        result = adjust_chart(fig, 2, 1, {"tickformat": "%b %Y"})
        assert result.layout.xaxis.tickformat == "%b %Y"

    def test_grid_config_applied_to_both_axes(self):
        fig = self._fig()
        result = adjust_chart(
            fig, 2, 1, {"grid": {"gridcolor": "lightgrey", "showgrid": True}}
        )
        assert result.layout.xaxis.gridcolor == "lightgrey"
        assert result.layout.yaxis.gridcolor == "lightgrey"

    def test_annotation_offset_default_shifts_x_only(self):
        fig = self._fig()
        original_x = [a.x for a in fig.layout.annotations]
        result = adjust_chart(fig, 2, 1, {})
        for orig_x, annotation in zip(original_x, result.layout.annotations):
            assert annotation.x == pytest.approx(orig_x + (1 / 2) / (2 + 0))

    def test_annotation_offset_config_shifts_x_and_y(self):
        fig = self._fig()
        original_x = [a.x for a in fig.layout.annotations]
        original_y = [a.y for a in fig.layout.annotations]
        result = adjust_chart(
            fig, 2, 1, {"annotations": {"x_offset": 1, "y_offset": 0.1}}
        )
        for ox, oy, annotation in zip(
            original_x, original_y, result.layout.annotations
        ):
            assert annotation.x == pytest.approx(ox + (1 / 2) / (2 + 1))
            assert annotation.y == pytest.approx(oy + 0.1)

    def test_extra_annotation_config_forwarded(self):
        fig = self._fig()
        result = adjust_chart(
            fig, 2, 1, {"annotations": {"font": {"size": 20}}}
        )
        for annotation in result.layout.annotations:
            assert annotation.font.size == 20

    def test_annotations_config_dict_is_not_mutated(self):
        # `annotations_config = config.get("annotations", {}).copy()` copies
        # before popping "x_offset"/"y_offset", so the caller's config dict
        # is left intact and can be reused across calls.
        annotations_cfg = {"x_offset": 1, "y_offset": 0.5}
        config = {"annotations": annotations_cfg}
        adjust_chart(self._fig(), 2, 1, config)
        assert annotations_cfg == {"x_offset": 1, "y_offset": 0.5}

    def test_returns_same_figure_instance(self):
        fig = self._fig()
        result = adjust_chart(fig, 2, 1, {})
        assert result is fig


# ---------------------------------------------------------------------------
# make_charts
# ---------------------------------------------------------------------------


class TestMakeCharts:
    def _data(self):
        return pl.DataFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                " __ col_a __Number of unique values": [1, 2, 3],
                "e__ col_a __Mean": [1.5, 2.5, 3.5],
            }
        )

    def test_writes_image_with_default_format_and_scale(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            Figure,
            "write_image",
            lambda self, file_path, format, scale: calls.append(
                (file_path, format, scale)
            ),
        )
        make_charts(self._data(), [(None, None), (None, None)], {}, "out.png")
        assert calls == [("out.png", "png", 1)]

    def test_writes_image_with_custom_format_and_scale(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            Figure,
            "write_image",
            lambda self, file_path, format, scale: calls.append(
                (file_path, format, scale)
            ),
        )
        make_charts(
            self._data(),
            [(None, None), (None, None)],
            {"format": "svg", "scale_factor": 2},
            "out.svg",
        )
        assert calls == [("out.svg", "svg", 2)]

    def test_one_data_column_still_creates_two_subplot_slots(self, monkeypatch):
        # n_subplots = max(2, data.shape[1] - 1), so a single stat column
        # still allocates a 2-slot (1x2) grid, leaving one subplot empty.
        captured = {}
        monkeypatch.setattr(
            Figure,
            "write_image",
            lambda self, file_path, format, scale: captured.setdefault("fig", self),
        )
        data = pl.DataFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1), date(2024, 1, 2)],
                " __ col_a __Number of unique values": [1, 2],
            }
        )
        make_charts(data, [(None, None)], {}, "out.png")
        fig = captured["fig"]
        assert len(fig.layout.annotations) == 1  # only one real subplot title
        assert len(fig.data) == 1  # only one trace added

    def test_outlier_bounds_produce_shapes_on_matching_subplot(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            Figure,
            "write_image",
            lambda self, file_path, format, scale: captured.setdefault("fig", self),
        )
        make_charts(
            self._data(),
            [(1.0, 2.5), (None, None)],
            {"outliers": {"style": {"fillcolor": "red"}}},
            "out.png",
        )
        fig = captured["fig"]
        assert len(fig.layout.shapes) == 2
        assert all(s.fillcolor == "red" for s in fig.layout.shapes)

    def test_insufficient_bounds_is_silently_swallowed_by_exception_handler(
        self, monkeypatch
    ):
        # BUG-ish: make_charts is decorated with @exception_handler(), which
        # catches ALL exceptions (logs and returns args[0] instead of
        # raising). Passing fewer `bounds` entries than data columns causes
        # an IndexError internally, but the caller sees no error at all —
        # just the original `data` DataFrame handed back and no file written.
        write_calls = []
        monkeypatch.setattr(
            Figure,
            "write_image",
            lambda self, file_path, format, scale: write_calls.append(file_path),
        )
        result = make_charts(self._data(), [(None, None)], {}, "out.png")
        assert result is not None  # actually the original `data` arg, not None
        assert write_calls == []  # write_image was never reached

    def test_missing_config_sections_use_defaults(self, monkeypatch):
        # Smoke test: an empty config dict must not raise, relying on the
        # `.get(..., {})` defaults threaded through every helper.
        monkeypatch.setattr(
            Figure, "write_image", lambda self, file_path, format, scale: None
        )
        result = make_charts(self._data(), [(None, None), (None, None)], {}, "out.png")
        assert result is None  # normal (non-exception) path returns None
