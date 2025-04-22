#!/usr/bin/env python3
"""
generate_report.py
Create a paginated HTML report (20 rows/page) from top_N.tsv, inlining
the highlighted SVGs for each RNA pair in two extra columns.

Usage:
  python3 generate_report.py \
    --pairs top_N.tsv \
    --svg-dir structures_plots/individual_svgs \
    --output report.html
"""
import base64
import pathlib
import argparse
import pandas as pd
from jinja2 import Template

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RNA‑pair report</title>
  <!-- DataTables -->
  <link  href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css" rel="stylesheet">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <style>
      body{ font-family: sans-serif; margin:1rem;}
      table.dataTable tbody td{ vertical-align:top; }
      .svg-cell{ min-width:260px; }
  </style>
</head>
<body>
  <h1>Top‑N structure pairs</h1>
  <table id="report" class="display">
    <thead>
      <tr>
        {% for col in cols %}<th>{{ col }}</th>{% endfor %}
        <th>Structure 1</th>
        <th>Structure 2</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        {% for col in cols %}<td>{{ row[col] }}</td>{% endfor %}
        <td class="svg-cell">{{ row["__svg1"]|safe }}</td>
        <td class="svg-cell">{{ row["__svg2"]|safe }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
<script>
$(document).ready(function(){
    $('#report').DataTable({ pageLength: 20 });
});
</script>
</body>
</html>
"""

def inline_svg(path: pathlib.Path) -> str:
    try:
        text = path.read_text()
        if text.lstrip().startswith('<?xml'):
            text = text.split('?>',1)[1]
        b64 = base64.b64encode(text.encode('utf-8')).decode('ascii')
        return f'<img src="data:image/svg+xml;base64,{b64}" width="250"/>'
    except Exception:
        return f'<span style="color:red">Missing {path.name}</span>'

def make_report(pairs_tsv, svg_dir, output_html):
    df = pd.read_csv(pairs_tsv, sep='\t')
    svg_dir = pathlib.Path(svg_dir)
    rows = []
    for idx, rec in df.iterrows():
        i = idx + 1
        # match the naming from draw_pairs.py
        id1 = rec.get('exon_id_1') or rec.get('id1')
        id2 = rec.get('exon_id_2') or rec.get('id2')
        safe1 = ''.join(c if c.isalnum() else '_' for c in f"{id1}_{i}")
        safe2 = ''.join(c if c.isalnum() else '_' for c in f"{id2}_{i}")
        rec = rec.to_dict()
        rec["__svg1"] = inline_svg(svg_dir / f"{safe1}.svg")
        rec["__svg2"] = inline_svg(svg_dir / f"{safe2}.svg")
        rows.append(rec)

    html = Template(HTML_TEMPLATE).render(
        cols=list(df.columns),
        rows=rows
    )
    pathlib.Path(output_html).write_text(html, encoding='utf-8')
    print(f"[ok] wrote {output_html}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--pairs',    required=True, help='top_N.tsv')
    p.add_argument('--svg-dir',  required=True, help='individual_svgs directory')
    p.add_argument('--output',   required=True, help='report.html path')
    args = p.parse_args()
    make_report(args.pairs, args.svg_dir, args.output)
