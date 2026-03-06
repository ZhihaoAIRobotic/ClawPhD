clawphd agent -m "Use the pdf_to_markdown tool to convert ./test.pdf into structured Markdown and editable figure assets.

Required parameters:
- pdf_path: ./test.pdf
- out_root: outputs/pdf2md
- backend: docling
- export_figures: true
- figure_box_source: auto
- export_svg: true
- export_drawio: true
- enable_rebuild: true
- rebuild_backend: auto
- rebuild_timeout_sec: 300

After completion, return:
1) output directory
2) markdown path
3) figures_total / svg_exported / drawio_exported / rebuilt_exported
4) warnings list"
