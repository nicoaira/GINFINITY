#!/usr/bin/env python3
"""
draw_pairs.py – generate RNArtistCore plots for RNA pairs in parallel,
highlight a window, strip the Reactivity legend, stitch SVGs side‑by‑side,
and export PNGs.

Usage:
  python3 draw_pairs.py --tsv top_N.tsv --outdir outdir \
        [--width W] [--height H] [--highlight-colour COLOUR] \
        [--num-workers N] [--keep-temp] [--debug]

Dependencies:
  • rnartistcore (Java jar alias)
  • Python 3.8+
  • pip install cairosvg
"""
import os, csv, subprocess, tempfile, shutil, xml.etree.ElementTree as ET
import argparse, sys, re, threading, concurrent.futures

# ensure SVG namespace
ET.register_namespace('', 'http://www.w3.org/2000/svg')

# load CairoSVG
try:
    import cairosvg
except ImportError:
    sys.exit("[fatal] Please install 'cairosvg' (pip install cairosvg)")

def parse_args():
    p = argparse.ArgumentParser(
        description='Draw & highlight RNA pairs – PNG output (parallel)')
    p.add_argument('--tsv',            required=True,  help='Input TSV')
    p.add_argument('--outdir',         required=True,  help='Output directory')
    p.add_argument('--width',    type=float, default=500,  help='SVG width')
    p.add_argument('--height',   type=float, default=500,  help='SVG height')
    p.add_argument('--highlight-colour', default="#FFD700",
                   help='HTML colour for window highlight')
    p.add_argument('--num-workers', type=int, default=1,
                   help='Parallel workers for rendering')
    p.add_argument('--keep-temp', action='store_true', help='Keep temp dir')
    p.add_argument('--debug',     action='store_true', help='Verbose')
    return p.parse_args()

# ─────────────────── RNArtist KTS template ───────────────────
def make_highlight(start, end, colour):
    if not start or not end:
        return ""
    return f'''        color {{
            value = "{colour}"
            location {{ {start} to {end} }}
        }}
'''

KTS_TEMPLATE = '''import io.github.fjossinet.rnartist.core.*

rnartist {{
    ss {{
        bn {{
            seq   = "{seq}"
            value = "{dot}"
            name  = "{name}"
        }}
    }}

    theme {{
        details {{ value = 4 }}
{highlight_block}    }}

    svg {{
        path   = "{path}"
        width  = {w}
        height = {h}
    }}
}}'''

# ─────────────────────────────────────────────────────────────
def combine(svg1, svg2, dst_svg):
    """Merge two SVGs side‑by‑side."""
    try:
        t1, t2 = ET.parse(svg1), ET.parse(svg2)
        r1, r2 = t1.getroot(), t2.getroot()
        w1 = float(r1.get('width','0').rstrip('px')); h1 = float(r1.get('height','0').rstrip('px'))
        w2 = float(r2.get('width','0').rstrip('px')); h2 = float(r2.get('height','0').rstrip('px'))
        W, H = w1 + w2, max(h1, h2)

        svg = ET.Element('{http://www.w3.org/2000/svg}svg',
                         {'width':str(W),'height':str(H)})
        g1 = ET.SubElement(svg,'g')
        for el in list(r1): g1.append(el)
        g2 = ET.SubElement(svg,'g',{'transform':f'translate({w1},0)'})
        for el in list(r2): g2.append(el)
        ET.ElementTree(svg).write(dst_svg, xml_declaration=True, encoding='utf-8')
        return True,(W,H)
    except Exception as e:
        print(f"[combine] {e}", file=sys.stderr)
        return False,(0,0)

def svg_to_png(src, dst):
    """Convert SVG → PNG via CairoSVG."""
    try:
        cairosvg.svg2png(url=src, write_to=dst)
        return True
    except Exception as e:
        print(f"[png] {e}", file=sys.stderr)
        return False

# strip RNArtistCore’s built‑in Reactivity legend
SVG_REACTIVITY_PATTERN = re.compile(
    r'<defs>.*?id="reactivities_scale".*?</text>\s*</svg>',
    flags=re.DOTALL
)
def strip_reactivity_legend(svg_path):
    try:
        txt = open(svg_path,'r',encoding='utf-8').read()
        cleaned, n = SVG_REACTIVITY_PATTERN.subn('</svg>', txt, count=1)
        if n:
            open(svg_path,'w',encoding='utf-8').write(cleaned)
    except Exception as e:
        print(f"[strip] {svg_path}: {e}", file=sys.stderr)

def validate(seq, dot):
    return len(seq)==len(dot)

# ─────────────────────────── main ───────────────────────────
def main():
    args   = parse_args()
    tsv    = os.path.abspath(args.tsv)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # prepare
    kts_dir = os.path.join(outdir,"kts_scripts"); os.makedirs(kts_dir, exist_ok=True)
    log_fh  = open(os.path.join(outdir,'failures.log'),'w')
    tmpdir  = tempfile.mkdtemp(prefix='rn_')
    print(f"[info] temp dir {tmpdir}")

    # load all rows
    with open(tsv) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        rows   = list(reader)

    log_lock = threading.Lock()

    def process_pair(i, row):
        # IDs & structures
        id1 = row.get('exon_id_1') or row.get('id1') or f"RNA_{i}_1"
        id2 = row.get('exon_id_2') or row.get('id2') or f"RNA_{i}_2"
        seq1 = row.get('sequence_1') or row.get('seq1') or ""
        seq2 = row.get('sequence_2') or row.get('seq2') or ""
        dot1 = row.get('secondary_structure_1') or row.get('structure1') or "."*len(seq1)
        dot2 = row.get('secondary_structure_2') or row.get('structure2') or "."*len(seq2)

        # convert T→U
        seq1 = seq1.replace('T','U').replace('t','u')
        seq2 = seq2.replace('T','U').replace('t','u')

        if not (validate(seq1,dot1) and validate(seq2,dot2)):
            return

        # highlight coords
        w1s = int(row.get('window_start_1',0) or 0)
        w1e = int(row.get('window_end_1',  0) or 0)
        w2s = int(row.get('window_start_2',0) or 0)
        w2e = int(row.get('window_end_2',  0) or 0)
        hl1 = make_highlight(w1s,w1e,args.highlight_colour)
        hl2 = make_highlight(w2s,w2e,args.highlight_colour)

        # safe names
        n1 = re.sub(r'[^A-Za-z0-9_]', '_', f"{id1}_{i}")
        n2 = re.sub(r'[^A-Za-z0-9_]', '_', f"{id2}_{i}")

        # doubles
        w_str = f"{args.width:.1f}"
        h_str = f"{args.height:.1f}"

        # KTS scripts
        s1 = KTS_TEMPLATE.format(path=tmpdir, w=w_str, h=h_str,
                                 seq=seq1, dot=dot1, name=n1,
                                 highlight_block=hl1)
        s2 = KTS_TEMPLATE.format(path=tmpdir, w=w_str, h=h_str,
                                 seq=seq2, dot=dot2, name=n2,
                                 highlight_block=hl2)

        # write
        for name,script in ((n1,s1),(n2,s2)):
            with open(os.path.join(kts_dir,f"{name}.kts"),'w') as fh: fh.write(script)
            with open(os.path.join(tmpdir,f"{name}.kts"),'w') as fh: fh.write(script)

        # draw
        if args.debug: print(f"[rnartist] {n1}")
        subprocess.run(['rnartistcore', os.path.join(tmpdir,f"{n1}.kts")])
        if args.debug: print(f"[rnartist] {n2}")
        subprocess.run(['rnartistcore', os.path.join(tmpdir,f"{n2}.kts")])

        svg1 = os.path.join(tmpdir, f"{n1}.svg")
        svg2 = os.path.join(tmpdir, f"{n2}.svg")
        if not (os.path.exists(svg1) and os.path.exists(svg2)):
            with log_lock: log_fh.write(f"{n1}\n{n2}\n")
            return

        # strip legends
        strip_reactivity_legend(svg1)
        strip_reactivity_legend(svg2)

        # copy individuals
        indiv = os.path.join(outdir,"individual_svgs"); os.makedirs(indiv, exist_ok=True)
        shutil.copy(svg1, os.path.join(indiv,f"{n1}.svg"))
        shutil.copy(svg2, os.path.join(indiv,f"{n2}.svg"))

        # combine & strip
        base    = f"pair_{i}_{id1}_{id2}"
        pair_svg= os.path.join(outdir, base + ".svg")
        ok,_    = combine(svg1,svg2,pair_svg)
        if ok: strip_reactivity_legend(pair_svg)

        # png
        pair_png = os.path.join(outdir, base + ".png")
        if svg_to_png(pair_svg, pair_png):
            print(f"[ok] {pair_png}")
        else:
            with log_lock: log_fh.write(f"{n1}\n{n2}\n")

    # dispatch
    if args.num_workers>1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as exe:
            futures = [exe.submit(process_pair,i,row) for i,row in enumerate(rows,1)]
            for _ in concurrent.futures.as_completed(futures): pass
    else:
        for i,row in enumerate(rows,1):
            process_pair(i,row)

    # cleanup
    log_fh.close()
    if args.keep_temp:
        print(f"[info] kept temp dir {tmpdir}")
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if args.debug: print(f"[info] removed temp dir")

if __name__ == '__main__':
    main()
