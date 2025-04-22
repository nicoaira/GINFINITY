#!/usr/bin/env python3
"""
draw_pairs.py – generate RNArtistCore plots for RNA pairs, highlight a window,
stitch them side‑by‑side, and export PNGs (SVGs are kept too).

Dependencies
------------
• RNArtistCore binary accessible as `rnartistcore`
• Python 3.8+
• pip install cairosvg
"""
import os, csv, subprocess, tempfile, shutil, xml.etree.ElementTree as ET
import argparse, sys, re

try:
    import cairosvg
except ImportError:
    sys.exit("[fatal] Please install 'cairosvg'  (pip install cairosvg)")

ET.register_namespace('', 'http://www.w3.org/2000/svg')

# ─────────────────────────── CLI ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Draw & highlight RNA pairs – PNG output')
    p.add_argument('--tsv', required=True, help='Input TSV with RNA pairs')
    p.add_argument('--outdir', required=True, help='Destination directory')
    p.add_argument('--width', type=float, default=500, help='Width of each RNA panel')
    p.add_argument('--height', type=float, default=500, help='Height of each RNA panel')
    p.add_argument('--highlight-colour', default="#FFD700",
                   help='Colour for highlighted window (hex or HTML name)')
    p.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    p.add_argument('--debug', action='store_true', help='Verbose output')
    return p.parse_args()

# ────────────────────── helpers / templates ─────────────────
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
        scheme  {{ value = "Persian Carolina" }}
{highlight_block}    }}

    svg {{
        path   = "{path}"
        width  = {w}
        height = {h}
    }}
}}'''

def combine(svg1, svg2, dst_svg):
    """side‑by‑side SVG merge; returns (ok,(W,H))."""
    try:
        t1, t2 = ET.parse(svg1), ET.parse(svg2)
        r1, r2 = t1.getroot(), t2.getroot()
        w1, h1 = float(r1.get('width','0').rstrip('px')), float(r1.get('height','0').rstrip('px'))
        w2, h2 = float(r2.get('width','0').rstrip('px')), float(r2.get('height','0').rstrip('px'))
        W, H   = w1 + w2, max(h1, h2)

        svg = ET.Element('{http://www.w3.org/2000/svg}svg',
                        {'width': str(W), 'height': str(H)})
        
        g1 = ET.SubElement(svg,'g'); [g1.append(el) for el in list(r1)]
        g2 = ET.SubElement(svg,'g',{'transform': f'translate({w1},0)'}); [g2.append(el) for el in list(r2)]
        ET.ElementTree(svg).write(dst_svg, xml_declaration=True, encoding='utf-8')
        return True,(W,H)
    except Exception as e:
        print(f"[combine] {e}", file=sys.stderr)
        return False,(0,0)

def svg_to_png(src, dst):
    try:
        cairosvg.svg2png(url=src, write_to=dst)
        return True
    except Exception as e:
        print(f"[png] {e}", file=sys.stderr)
        return False

def validate(seq, dot): return len(seq)==len(dot)

# ─────────────────────────── main ───────────────────────────
def main():
    args   = parse_args()
    tsv    = os.path.abspath(args.tsv)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    kts_dir = os.path.join(outdir,"kts_scripts"); os.makedirs(kts_dir, exist_ok=True)
    log_fh  = open(os.path.join(outdir,'failures.log'),'w')

    tmpdir = tempfile.mkdtemp(prefix='rn_')
    print(f"[info] temp dir {tmpdir}")

    try:
        with open(tsv) as fh:
            rdr  = csv.DictReader(fh, delimiter='\t')
            cols = rdr.fieldnames or []

            # identify columns
            id_cols     = [c for c in ('exon_id_1','exon_id_2','id1','id2') if c in cols]
            seq_cols    = [c for c in ('exon_sequence_1','exon_sequence_2','seq1','seq2') if c in cols]
            struct_cols = [c for c in ('ss_fine_tuned_1','ss_fine_tuned_2','structure1','structure2') if c in cols]
            if not id_cols or not seq_cols or not struct_cols:
                print("[fatal] TSV must have ID/sequence/structure columns", file=sys.stderr); return

            id1_c,id2_c       = id_cols[0], id_cols[-1] if len(id_cols)>1 else id_cols[0]
            seq1_c,seq2_c     = seq_cols[0], seq_cols[-1] if len(seq_cols)>1 else seq_cols[0]
            struct1_c,struct2_c = struct_cols[0], struct_cols[-1] if len(struct_cols)>1 else struct_cols[0]

            for i,row in enumerate(rdr,1):
                id1,id2 = row.get(id1_c,f"RNA_{i}_1"), row.get(id2_c,f"RNA_{i}_2")
                seq1,seq2 = row.get(seq1_c,""), row.get(seq2_c,"")
                dot1,dot2 = row.get(struct1_c,"."*len(seq1)), row.get(struct2_c,"."*len(seq2))
                if not (validate(seq1,dot1) and validate(seq2,dot2)):
                    print(f"[warn] len mismatch in pair {i}", file=sys.stderr); continue

                w1s,w1e = int(row.get('window_start_1',0) or 0), int(row.get('window_end_1',0) or 0)
                w2s,w2e = int(row.get('window_start_2',0) or 0), int(row.get('window_end_2',0) or 0)
                hl1 = make_highlight(w1s,w1e,args.highlight_colour)
                hl2 = make_highlight(w2s,w2e,args.highlight_colour)

                # RNArtist needs name chars [A‑Z,a‑z,0‑9,_]
                n1 = re.sub(r'[^A-Za-z0-9_]', '_', f"{id1}_{i}")
                n2 = re.sub(r'[^A-Za-z0-9_]', '_', f"{id2}_{i}")

                # ensure Double literals for width/height
                w_str,h_str = f"{args.width:.1f}", f"{args.height:.1f}"

                s1 = KTS_TEMPLATE.format(path=tmpdir,w=w_str,h=h_str,
                                         seq=seq1,dot=dot1,name=n1,highlight_block=hl1)
                s2 = KTS_TEMPLATE.format(path=tmpdir,w=w_str,h=h_str,
                                         seq=seq2,dot=dot2,name=n2,highlight_block=hl2)

                # write scripts (for debug and execution)
                for name,script in ((n1,s1),(n2,s2)):
                    open(os.path.join(kts_dir,f"{name}.kts"),'w').write(script)
                    open(os.path.join(tmpdir,f"{name}.kts"),'w').write(script)

                # run rnartistcore
                if args.debug: print(f"[rnartist] {n1}")
                r1 = subprocess.run(['rnartistcore', os.path.join(tmpdir,f"{n1}.kts")])
                if args.debug: print(f"[rnartist] {n2}")
                r2 = subprocess.run(['rnartistcore', os.path.join(tmpdir,f"{n2}.kts")])

                svg1,svg2 = os.path.join(tmpdir,f"{n1}.svg"), os.path.join(tmpdir,f"{n2}.svg")
                if not (os.path.exists(svg1) and os.path.exists(svg2)):
                    log_fh.write(f"{n1}\n{n2}\n"); continue

                # keep individual SVGs
                indiv = os.path.join(outdir,"individual_svgs"); os.makedirs(indiv, exist_ok=True)
                shutil.copy(svg1, os.path.join(indiv,f"{n1}.svg"))
                shutil.copy(svg2, os.path.join(indiv,f"{n2}.svg"))

                base = f"pair_{i}_{id1}_{id2}"
                pair_svg = os.path.join(outdir, base + ".svg")
                ok,_ = combine(svg1,svg2,pair_svg)
                if not ok:
                    log_fh.write(f"{n1}\n{n2}\n"); continue

                pair_png = os.path.join(outdir, base + ".png")
                if svg_to_png(pair_svg,pair_png):
                    print(f"[ok] {pair_png}")
                else:
                    log_fh.write(f"{n1}\n{n2}\n")
    finally:
        log_fh.close()
        if args.keep_temp:
            print(f"[info] kept temp dir {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if args.debug: print(f"[info] removed temp dir")

if __name__ == "__main__":
    main()
