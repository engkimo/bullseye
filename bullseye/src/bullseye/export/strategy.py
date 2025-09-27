from __future__ import annotations

from typing import Any

from ..core.interfaces import ExporterIF
from . import export_json as _export_json
from . import export_html as _export_html
from . import export_markdown as _export_markdown
from . import export_csv as _export_csv


class JSONExporter(ExporterIF):
    def export(self, document: Any, out_path: str, **options: Any) -> str:
        _export_json.export_json(
            document,
            out_path,
            ignore_line_break=bool(options.get('ignore_line_break', False)),
            encoding=str(options.get('encoding', 'utf-8')),
            img=options.get('img'),
            export_figure=bool(options.get('export_figure', False)),
            figure_dir=str(options.get('figure_dir', 'figures')),
        )
        return out_path


class HTMLExporter(ExporterIF):
    def export(self, document: Any, out_path: str, **options: Any) -> str:
        _export_html.export_html(
            document,
            out_path,
            ignore_line_break=bool(options.get('ignore_line_break', False)),
            img=options.get('img'))
        return out_path


class MarkdownExporter(ExporterIF):
    def export(self, document: Any, out_path: str, **options: Any) -> str:
        _export_markdown.export_markdown(
            document,
            out_path,
            ignore_line_break=bool(options.get('ignore_line_break', False)))
        return out_path


class CSVExporter(ExporterIF):
    def export(self, document: Any, out_path: str, **options: Any) -> str:
        _export_csv.export_csv(
            document,
            out_path,
            encoding=str(options.get('encoding', 'utf-8')))
        return out_path


def get_exporter(fmt: str) -> ExporterIF:
    key = (fmt or '').strip().lower()
    if key in ('json', 'udj'):
        return JSONExporter()
    if key in ('html',):
        return HTMLExporter()
    if key in ('md', 'markdown'):
        return MarkdownExporter()
    if key in ('csv',):
        return CSVExporter()
    raise ValueError(f"Unknown export format: {fmt}")

