#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*─────────────────────────────────────────────────────────────
 *  USER-TUNABLE PARAMETERS  (all still overridable on CLI)
 *────────────────────────────────────────────────────────────*/
params.input                   = "$baseDir/input.tsv"          // TSV with sequences
params.outdir                  = "results"
params.model_path              = "$baseDir/model.pth"
params.structure_column_name   = 'secondary_structure'
params.structure_column_num    = null
params.id_column               = 'transcript_id'

params.device                  = 'cpu'
params.num_workers             = 4
params.subgraphs               = false
params.L                       = null
params.keep_paired_neighbors   = false
params.retries                 = 0

params.batch_size_embed        = 100        // full sequences per embedding task

// faiss index defaults
params.faiss_k = 500    

// compute-distances defaults
params.embedding_col           = 'embedding_vector'
params.keep_cols               = 'transcript_id,window_start,window_end'
params.num_workers_dist        = 1
params.device_dist             = 'cpu'
params.batch_size              = 4096
params.mode                    = 2
params.query                   = null

// aggregation defaults
params.alpha1                  = 0.25
params.alpha2                  = 0.24
params.beta1                   = 0.0057
params.beta2                   = 1.15
params.gamma                   = 0.41
params.percentile              = 1
params.top_n                   = 10

// plotting flags
params.plot_distances_distribution = true
params.hist_seed              = 42
params.hist_frac              = 0.001
params.hist_bins              = 200

params.plot_score_distribution = true
params.score_bins            = 30

/*─────────────────────────────────────────────────────────────
 *  BASIC CHECK
 *────────────────────────────────────────────────────────────*/
if ( !file(params.input).exists() )
    error "✘ Cannot find the input TSV:  ${params.input}"

/*─────────────────────────────────────────────────────────────
 *  BUILD BATCHES IN-MEMORY
 *────────────────────────────────────────────────────────────*/
def batch_ch = Channel
    .fromPath(params.input)
    .splitCsv(header: true, by: params.batch_size_embed)

/*─────────────────────────────────────────────────────────────
 *  1)  GENERATE EMBEDDINGS PER BATCH  (no more publish)
 *─────────────────────────────────────────────────────────────*/
process GENERATE_EMBEDDINGS {
    tag       { "generate_embeddings batch=${task.index}" }
    maxForks = 1

    input:
      val rows                               // list< Map >

    output:
      path "embeddings_batch_${task.index}.tsv", emit: batch_embeddings

    script:
    // ─ reconstruct TSV for this batch ─
    def header = new File(params.input).withReader { it.readLine() }
    def lines  = rows.collect { it.values().join('\t') }.join('\n')

    """
    cat > batch_${task.index}.tsv <<'EOF'
${header}
${lines}
EOF

    python3 ${baseDir}/modules/predict_embedding.py \
      --input batch_${task.index}.tsv \
      --id-column ${params.id_column} \
      --model_path ${params.model_path} \
      --output embeddings_batch_${task.index}.tsv \
      --keep-cols ${params.id_column},seq_len \
      --structure_column_name ${params.structure_column_name} \
      ${params.structure_column_num ? "--structure_column_num ${params.structure_column_num}" : ''} \
      --header true \
      --device ${params.device} \
      --num_workers ${params.num_workers} \
      ${params.subgraphs            ? '--subgraphs' : ''} \
      ${params.L                    ? "--L ${params.L}" : ''} \
      ${params.keep_paired_neighbors? '--keep_paired_neighbors' : ''} \
      --retries ${params.retries}
    """
}

/*─────────────────────────────────────────────────────────────
 *  2)  MERGE PER-BATCH EMBEDDINGS  (only this publishes)
 *─────────────────────────────────────────────────────────────*/
process MERGE_EMBEDDINGS {
    container ''                       // <<<< disable Docker for this step
    tag       "merge_embeddings"
    publishDir "${params.outdir}", mode:'copy'

    input:
      path batch_embeddings

    output:
      path "embeddings.tsv", emit: embeddings

    script:
    """
    head -n1 ${batch_embeddings[0]} > embeddings.tsv
    for f in ${batch_embeddings.join(' ')}; do
        tail -n +2 \$f >> embeddings.tsv
    done
    """
}

process BUILD_FAISS_INDEX {
    tag    "build_faiss_index"

    input:
      path embeddings

    output:
      path "faiss.index",       emit: faiss_idx
      path "faiss_mapping.tsv", emit: faiss_map

    script:
    """
    python3 ${baseDir}/modules/build_faiss_index.py \
      --input embeddings.tsv \
      --id-column ${params.id_column} \
      --query ${params.query} \
      --index-path faiss.index \
      --mapping-path faiss_mapping.tsv
    """
}

process QUERY_FAISS_INDEX {
    tag    "query_faiss_index"
    cpus   params.num_workers_dist

    input:
      path embeddings
      path faiss_idx
      path faiss_map

    output:
      path "distances.tsv", emit: distances

    script:
    """
    python3 ${baseDir}/modules/query_faiss_index.py \
      --input embeddings.tsv \
      --id-column ${params.id_column} \
      --query ${params.query} \
      --index-path faiss.index \
      --mapping-path faiss_mapping.tsv \
      --top-k ${params.faiss_k} \
      --output distances.tsv
    """
}

/*─────────────────────────────────────────────────────────────
 *  3)  COMPUTE DISTANCES
 *────────────────────────────────────────────────────────────*/
process COMPUTE_DISTANCES {
    tag       "compute_distances"
    publishDir "${params.outdir}", mode:'copy'
    cpus      params.num_workers_dist

    input:
      path embeddings

    output:
      path "distances.tsv", emit: distances

    script:
    """
    python3 ${baseDir}/modules/compute_distances.py \
      --input embeddings.tsv \
      --output distances.tsv \
      --embedding-col ${params.embedding_col} \
      --keep-cols ${params.keep_cols} \
      --num-workers ${params.num_workers_dist} \
      --device ${params.device_dist} \
      --batch-size ${params.batch_size} \
      --mode ${params.mode} \
      --id-column ${params.id_column} \
      --query ${params.query}
    """
}

/*─────────────────────────────────────────────────────────────
 *  4)  SORT DISTANCES
 *────────────────────────────────────────────────────────────*/
process SORT_DISTANCES {
    tag       "sort_distances"
    publishDir "${params.outdir}", mode:'copy'

    input:
      path distances

    output:
      path "distances.sorted.tsv", emit: sorted_distances

    script:
    """
    python3 - <<'PY'
import pandas as pd
df = pd.read_csv('${distances}', sep='\\t')
df.sort_values('distance', inplace=True)
df.to_csv('distances.sorted.tsv', sep='\\t', index=False)
PY
    """
}

/*─────────────────────────────────────────────────────────────
 *  5)  (OPTIONAL) PLOT DISTANCES
 *────────────────────────────────────────────────────────────*/
process PLOT_DISTANCES {
    when   { params.plot_distances_distribution }
    tag    "plot_distances"
    publishDir "${params.outdir}/plots", mode:'copy'

    input:
      path sorted_distances

    output:
      path "distance_distribution.png"

    script:
    """
    python3 - <<'PY'
import pandas as pd, numpy as np, matplotlib.pyplot as plt
seed=${params.hist_seed}; frac=${params.hist_frac}; bins=${params.hist_bins}
df=pd.read_csv('${sorted_distances}',sep='\\t')
sample=df.sample(frac=frac,random_state=seed)
plt.figure(figsize=(8,5))
plt.hist(sample['distance'],bins=bins)
plt.xlabel('Distance'); plt.ylabel('Frequency')
plt.title(f'Distance Distribution ({frac*100:.1f}% sample)')
plt.tight_layout(); plt.savefig('distance_distribution.png')
PY
    """
}

/*─────────────────────────────────────────────────────────────
 *  6)  AGGREGATE + ENRICH
 *────────────────────────────────────────────────────────────*/
process AGGREGATE_SCORE {
    tag       "aggregate_score"
    publishDir "${params.outdir}", mode:'copy'
    cpus      params.num_workers

    input:
      path sorted_distances
      path input_tsv

    output:
      path "pairs_scores_all_contigs.tsv",              emit: enriched_all
      path "pairs_scores_all_contigs.unaggregated.tsv", emit: enriched_unagg

    // ← updated script section:
    script:
    """
    # 1) run the core aggregation
    python3 ${baseDir}/modules/aggregated_score.py \
      --input ${sorted_distances} \
      --id-column ${params.id_column} \
      --alpha1 ${params.alpha1} --alpha2 ${params.alpha2} \
      --beta1 ${params.beta1}   --beta2 ${params.beta2} \
      --gamma ${params.gamma}   --percentile ${params.percentile} \
      --mode contigs \
      --output raw_contigs.tsv \
      --output-unaggregated

    # 2) enrich with whatever columns exist in the original metadata
    python3 - << 'PY'
import pandas as pd, sys, pathlib, os

# ------- inputs -------
idc      = '${params.id_column}'
meta_tsv = pathlib.Path('${input_tsv}')
df_all   = pd.read_csv('raw_contigs.tsv',            sep='\\t')
df_un    = pd.read_csv('raw_contigs.unaggregated.tsv', sep='\\t')
# ----------------------

meta = pd.read_csv(meta_tsv, sep='\\t', dtype=str)

# which optional columns are actually present?
optional_cols = ['gene_name', 'exon_sequence', '${params.structure_column_name}']
present_cols  = [c for c in optional_cols if c in meta.columns]

# build two mapping DataFrames with *only* the present columns
m1 = meta[[idc] + present_cols].rename(
        columns={ idc:f'{idc}_1', **{c: f'{c}_1' for c in present_cols} })
m2 = meta[[idc] + present_cols].rename(
        columns={ idc:f'{idc}_2', **{c: f'{c}_2' for c in present_cols} })

# left-merge; missing cols simply stay absent
df_all = df_all.merge(m1, on=f'{idc}_1', how='left').merge(m2, on=f'{idc}_2', how='left')
df_un  = df_un .merge(m1, on=f'{idc}_1', how='left').merge(m2, on=f'{idc}_2', how='left')

df_all.to_csv('pairs_scores_all_contigs.tsv',               sep='\t', index=False)
df_un .to_csv('pairs_scores_all_contigs.unaggregated.tsv',  sep='\t', index=False)
PY
    """
}

/*─────────────────────────────────────────────────────────────
 *  7)  (OPTIONAL) PLOT SCORE
 *────────────────────────────────────────────────────────────*/
process PLOT_SCORE {
    when   { params.plot_score_distribution }
    tag    "plot_score"
    publishDir "${params.outdir}/plots", mode:'copy'

    input:
      path enriched_all

    output:
      path "score_distribution.png"

    script:
    """
    python3 - <<'PY'
import pandas as pd, matplotlib.pyplot as plt
bins=${params.score_bins}
df=pd.read_csv('${enriched_all}',sep='\\t')
plt.figure(figsize=(8,5))
plt.hist(df['score'],bins=bins)
plt.xlabel('Score'); plt.ylabel('Frequency')
plt.title('Contig Score Distribution')
plt.tight_layout(); plt.savefig('score_distribution.png')
PY
    """
}

/*─────────────────────────────────────────────────────────────
 *  8)  FILTER TOP-N CONTIGS (+ their windows)
 *────────────────────────────────────────────────────────────*/
process FILTER_TOP_CONTIGS {
    tag       "filter_top_contigs"
    publishDir "${params.outdir}", mode:'copy'

    input:
      path enriched_all
      path enriched_unagg

    output:
      path "pairs_scores_top_contigs.tsv",              emit: top_contigs
      path "pairs_scores_top_contigs.unaggregated.tsv", emit: top_contigs_unagg

    script:
    """
    python3 - <<'PY'
import pandas as pd
top=${params.top_n}
all=pd.read_csv('${enriched_all}',sep='\\t')
un=pd.read_csv('${enriched_unagg}',sep='\\t')
sel=all[all['contig_rank']<=top]
ids=sel['contig_id'].unique()
sel_un=un[un['contig_id'].isin(ids)]
sel.to_csv('pairs_scores_top_contigs.tsv',sep='\\t',index=False)
sel_un.to_csv('pairs_scores_top_contigs.unaggregated.tsv',sep='\\t',index=False)
PY
    """
}

/*─────────────────────────────────────────────────────────────
 *  9)  DRAW CONTIG-LEVEL SVGs + PNGs
 *────────────────────────────────────────────────────────────*/
process DRAW_CONTIG_SVGS {
    tag       "draw_contig_svgs"
    publishDir "${params.outdir}/drawings/contigs_drawings", mode:'copy'
    cpus      params.num_workers

    input:
      path top_contigs_tsv

    output:
      path 'failures.log',    emit: contig_failures
      path 'kts_scripts',     emit: contig_kts
      path 'individual_svgs', emit: contig_individual
      path 'pairs_drawings',  emit: contig_pairs

    script:
    """
    mkdir -p kts_scripts individual_svgs \
            pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
    touch failures.log

    python3 - <<'PY'
import pandas as pd
df=pd.read_csv('${top_contigs_tsv}',sep='\\t')
df=df.rename(columns={'contig_start_1':'window_start_1','contig_end_1':'window_end_1'})
df.to_csv('to_draw_contigs.tsv',sep='\\t',index=False)
PY

    python3 /app/draw_pairs.py \
      --tsv to_draw_contigs.tsv \
      --outdir . \
      --width 500 --height 500 \
      --highlight-colour "#00FF99" \
      --num-workers ${params.num_workers}

    mv pair_*.svg pairs_drawings/pairs_svgs/ 2>/dev/null || true
    mv pair_*.png pairs_drawings/pairs_pngs/ 2>/dev/null || true
    """
}

/*─────────────────────────────────────────────────────────────
 * 10)  DRAW WINDOW-LEVEL SVGs + PNGs
 *────────────────────────────────────────────────────────────*/
process DRAW_UNAGG_SVGS {
    tag       "draw_window_svgs"
    publishDir "${params.outdir}/drawings/unagg_windows_drawings", mode:'copy'
    cpus      params.num_workers

    input:
      path top_windows_tsv

    output:
      path 'failures.log',    emit: window_failures
      path 'kts_scripts',     emit: window_kts
      path 'individual_svgs', emit: window_individual
      path 'pairs_drawings',  emit: window_pairs

    script:
    """
    mkdir -p kts_scripts individual_svgs \
            pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
    touch failures.log

    cp ${top_windows_tsv} to_draw_windows.tsv

    python3 /app/draw_pairs.py \
      --tsv to_draw_windows.tsv \
      --outdir . \
      --width 500 --height 500 \
      --highlight-colour "#00FF99" \
      --num-workers ${params.num_workers}

    mv pair_*.svg pairs_drawings/pairs_svgs/ 2>/dev/null || true
    mv pair_*.png pairs_drawings/pairs_pngs/ 2>/dev/null || true
    """
}

/*─────────────────────────────────────────────────────────────
 * 11)  GENERATE CONTIG-LEVEL HTML REPORT
 *────────────────────────────────────────────────────────────*/
process GENERATE_AGGREGATED_REPORT {
    tag       "gen_agg_report"
    publishDir "${params.outdir}", mode:'copy'

    input:
      path top_contigs_tsv
      path contig_svgs

    output:
      path "pairs_contigs_report.html"

    script:
    """
    python3 ${baseDir}/modules/generate_report.py \
      --pairs ${top_contigs_tsv} \
      --svg-dir ${contig_svgs} \
      --id-column ${params.id_column} \
      --output pairs_contigs_report.html
    """
}

/*─────────────────────────────────────────────────────────────
 * 12)  GENERATE WINDOW-LEVEL HTML REPORT
 *────────────────────────────────────────────────────────────*/
process GENERATE_UNAGGREGATED_REPORT {
    tag       "gen_unagg_report"
    publishDir "${params.outdir}", mode:'copy'

    input:
      path top_windows_tsv
      path window_svgs

    output:
      path "pairs_contigs_report.unaggregated.html"

    script:
    """
    python3 ${baseDir}/modules/generate_report.py \
      --pairs ${top_windows_tsv} \
      --svg-dir ${window_svgs} \
      --id-column ${params.id_column} \
      --output pairs_contigs_report.unaggregated.html
    """
}

/*─────────────────────────────────────────────────────────────
 *  WORKFLOW  (connect everything)
 *────────────────────────────────────────────────────────────*/
workflow {

    /* 1-2  embeddings + merge */
    def gen    = GENERATE_EMBEDDINGS(batch_ch)
    def merged = MERGE_EMBEDDINGS(gen.batch_embeddings.collect())

    /* 3-4   build & query FAISS index instead of brute force */
    def idx       = BUILD_FAISS_INDEX(merged.embeddings)
    def distances = QUERY_FAISS_INDEX(merged.embeddings, idx.faiss_idx, idx.faiss_map)

    /* 5     sort + optional plot */
    def sorted_dist = SORT_DISTANCES(distances)
    PLOT_DISTANCES(sorted_dist)

    /* 6-7   aggregation & optional score plot */
    def (agg_all, agg_un) = AGGREGATE_SCORE(sorted_dist, Channel.fromPath(params.input))
    PLOT_SCORE(agg_all)

    /* 8     top-N selection */
    def (top_contigs, top_un) = FILTER_TOP_CONTIGS(agg_all, agg_un)

    /* 9-10  drawings */
    def contig_draws  = DRAW_CONTIG_SVGS(top_contigs)
    def window_draws  = DRAW_UNAGG_SVGS(top_un)

    /* 11-12 HTML reports */
    GENERATE_AGGREGATED_REPORT(top_contigs, contig_draws.contig_individual)
    GENERATE_UNAGGREGATED_REPORT(top_un, window_draws.window_individual)
}