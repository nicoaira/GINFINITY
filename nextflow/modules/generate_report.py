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
      /* Monospace for sequence columns */
      {% for col in sequence_cols %}td.{{col|replace(' ', '_')}}{ font-family: monospace; }
      {% endfor %}
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
        {% for col in cols %}
          <td class="{{col|replace(' ', '_')}}">
            {% if col in sequence_cols %}
              {{ row[col]|safe }}  {# Safe render for <br> tags #}
            {% else %}
              {{ row[col] }}
            {% endif %}
          </td>
        {% endfor %}
        <td class="svg-cell">{{ row["__svg1"]|safe }}</td>
        <td class="svg-cell">{{ row["__svg2"]|safe }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
<script>
$(document).ready(function(){
    $('#report').DataTable({
        pageLength: 20
        {% if metric_col_index is not none %},
        order: [[ {{ metric_col_index }}, 'desc' ]]
        {% endif %}
    });
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
    
    # Auto-detect sequence columns (case-insensitive)
    sequence_cols = [col for col in df.columns if 'sequence' in col.lower()]
    
    # Determine metric column index if exists
    metric_col_index = None
    if 'metric' in df.columns:
        metric_col_index = df.columns.get_loc('metric')
    
    rows = []
    for idx, rec in df.iterrows():
        i = idx + 1
        # Handle sequence columns
        row_data = rec.to_dict()
        for col in sequence_cols:
            if col in row_data:
                val = row_data[col]
                # Handle NaN and convert to string
                if pd.isna(val):
                    processed = ""
                else:
                    seq = str(val).strip()
                    # Split into 30-char chunks with <br>
                    chunks = [seq[i:i+30] for i in range(0, len(seq), 30)]
                    processed = '<br>'.join(chunks)
                row_data[col] = processed
        
        # Add SVG images
        id1 = row_data.get('exon_id_1') or row_data.get('id1')
        id2 = row_data.get('exon_id_2') or row_data.get('id2')
        safe1 = ''.join(c if c.isalnum() else '_' for c in f"{id1}_{i}")
        safe2 = ''.join(c if c.isalnum() else '_' for c in f"{id2}_{i}")
        row_data["__svg1"] = inline_svg(svg_dir / f"{safe1}.svg")
        row_data["__svg2"] = inline_svg(svg_dir / f"{safe2}.svg")
        
        rows.append(row_data)
    
    # Generate HTML
    html = Template(HTML_TEMPLATE).render(
        cols=list(df.columns),
        rows=rows,
        metric_col_index=metric_col_index,
        sequence_cols=sequence_cols
    )
    pathlib.Path(output_html).write_text(html, encoding='utf-8')
    print(f"[ok] Wrote report with formatted sequences to {output_html}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--pairs',    required=True, help='top_N.tsv')
    p.add_argument('--svg-dir',  required=True, help='Directory with SVG files')
    p.add_argument('--output',   required=True, help='Output HTML path')
    args = p.parse_args()
    make_report(args.pairs, args.svg_dir, args.output)