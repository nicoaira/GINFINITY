#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// — Top-level channel for the original input TSV
def input_tsv_ch = Channel.fromPath(params.input)

workflow {

    // 1) Generate embeddings
    def embeddings_ch = params.embeddings_file ?
        Channel.fromPath(params.embeddings_file) :
        GENERATE_EMBEDDINGS(input_tsv_ch)

    // 2) Compute raw distances
    def distances_ch = COMPUTE_DISTANCES(embeddings_ch)

    // 3) Sort distances ascending
    def sorted_distances_ch = SORT_DISTANCES(distances_ch)

    // 4) Aggregate + enrich contigs & windows metrics
    def (enriched_all_ch, enriched_unagg_ch) = AGGREGATE_METRIC(sorted_distances_ch, input_tsv_ch)

    // 5) Filter out the top-N contigs & windows
    def (top_contigs_ch, top_contigs_unagg_ch) = FILTER_TOP_CONTIGS(enriched_all_ch, enriched_unagg_ch)

    // 6) DRAW_CONTIG_SVGS emits 4 channels – destructure them:
    def (contig_failures_ch, contig_kts_ch, contig_individual_ch, contig_pairs_ch) = 
                                DRAW_CONTIG_SVGS(top_contigs_ch)

    // 7) DRAW_UNAGG_SVGS emits 4 channels – destructure as well:
    def (win_failures_ch, win_kts_ch, window_individual_ch, window_pairs_ch) =
                                DRAW_UNAGG_SVGS(top_contigs_unagg_ch)

    // 8a) Generate aggregated‐contig report: only needs
    //     - the top‐contigs TSV
    //     - the individual_svgs dir
    def agg_report_ch = GENERATE_AGGREGATED_REPORT(top_contigs_ch, contig_individual_ch)

    // 8b) Generate unaggregated‐window report: 
    //     - the top‐windows TSV
    //     - the individual_svgs dir for windows
    def unagg_report_ch = GENERATE_UNAGGREGATED_REPORT(top_contigs_unagg_ch, window_individual_ch)
}



process GENERATE_EMBEDDINGS {
    tag "generate_embeddings"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path input_file

    output:
      path "${params.outdir}/embeddings.tsv", emit: embeddings
      // change with  path "embeddings.tsv", emit: embeddings

    script:
    """
    # Add seq_len column
    python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('${input_file}', sep='\\t')
df['seq_len'] = df['exon_sequence'].str.len()
df.to_csv('with_seq_len.tsv', sep='\\t', index=False)
EOF

    # Generate embeddings
    python3 ${baseDir}/modules/predict_embedding.py \
      --input with_seq_len.tsv \
      --model_path ${params.model_path} \
      --output ${params.outdir}/embeddings.tsv \
      --structure_column_name ${params.structure_column_name} \
      ${params.structure_column_num  ? "--structure_column_num ${params.structure_column_num}" : ''} \
      --header ${params.header} \
      --device ${params.device} \
      --num_workers ${params.num_workers} \
      ${params.subgraphs            ? '--subgraphs' : ''} \
      ${params.L                    ? "--L ${params.L}" : ''} \
      ${params.keep_paired_neighbors? '--keep_paired_neighbors' : ''} \
      --retries ${params.retries}
    """
}


process COMPUTE_DISTANCES {
    tag "compute_distances"

    input:
      path embeddings

    output:
      path "distances.tsv", emit: distances

    script:
    """
    python3 ${baseDir}/modules/compute_distances.py --input ${embeddings} \
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


process SORT_DISTANCES {
    tag "sort_distances"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path distances

    output:
      path "distances.sorted.tsv", emit: sorted_distances

    script:
    """
    python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('${distances}', sep='\\t')
df.sort_values('distance', inplace=True)
df.to_csv('distances.sorted.tsv', sep='\\t', index=False)
EOF
    """
}


process AGGREGATE_METRIC {
    tag "aggregate_metric"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path sorted_distances
      path input_tsv

    output:
      // only the enriched outputs go to results/
      path "exon_pairs_scores_all_contigs.tsv",              emit: enriched_all
      path "exon_pairs_scores_all_contigs.unaggregated.tsv", emit: enriched_unagg

    script:
    """
    # 1) Run the core aggregation to get raw intermediate tables
    python3 ${baseDir}/modules/aggregated_metric.py \
      --input ${sorted_distances} \
      --alpha1 ${params.alpha1} \
      --alpha2 ${params.alpha2} \
      --beta1 ${params.beta1} \
      --beta2 ${params.beta2} \
      --gamma ${params.gamma} \
      --percentile ${params.percentile} \
      --mode contigs \
      --output raw_contigs.tsv \
      --output-unaggregated

    # 2) Immediately enrich and write final contig files
    python3 - << 'EOF'
import pandas as pd

df_all = pd.read_csv('raw_contigs.tsv', sep='\\t')
df_un  = pd.read_csv('raw_contigs.unaggregated.tsv', sep='\\t')
df_meta = pd.read_csv('${input_tsv}', sep='\\t', dtype=str)

idcol     = 'exon_id'
structcol = '${params.structure_column_name}'

m1 = df_meta[[idcol,'gene_name','exon_sequence', structcol]]\
      .rename(columns={
         idcol: 'exon_id_1',
         'gene_name': 'gene_name_1',
         'exon_sequence': 'sequence_1',
         structcol: 'secondary_structure_1'
      })

m2 = df_meta[[idcol,'gene_name','exon_sequence', structcol]]\
      .rename(columns={
         idcol: 'exon_id_2',
         'gene_name': 'gene_name_2',
         'exon_sequence': 'sequence_2',
         structcol: 'secondary_structure_2'
      })

# write enriched contig summary
df_all = df_all.merge(m1, on='exon_id_1', how='left')\
               .merge(m2, on='exon_id_2', how='left')
df_all.to_csv('exon_pairs_scores_all_contigs.tsv', sep='\\t', index=False)

# write enriched unaggregated windows
df_un = df_un.merge(m1, on='exon_id_1', how='left')\
             .merge(m2, on='exon_id_2', how='left')
df_un.to_csv('exon_pairs_scores_all_contigs.unaggregated.tsv', sep='\\t', index=False)
EOF
    """
}


process FILTER_TOP_CONTIGS {
    tag "filter_top_contigs"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path enriched_all
      path enriched_unagg

    output:
      path "exon_pairs_scores_top_contigs.tsv",               emit: top_contigs
      path "exon_pairs_scores_top_contigs.unaggregated.tsv", emit: top_contigs_unagg

    script:
    """
    python3 - << 'EOF'
import pandas as pd

# load enriched tables
df_all = pd.read_csv('${enriched_all}', sep='\\t')
df_un  = pd.read_csv('${enriched_unagg}', sep='\\t')

# pick top N contigs by contig_rank <= params.top_n
top_n = ${params.top_n}
df_top_all = df_all[df_all['contig_rank'] <= top_n]

# filter unaggregated windows to only those contigs
top_ids = df_top_all['contig_id'].unique().tolist()
df_top_un = df_un[df_un['contig_id'].isin(top_ids)]

# write results
df_top_all.to_csv('exon_pairs_scores_top_contigs.tsv', sep='\\t', index=False)
df_top_un.to_csv('exon_pairs_scores_top_contigs.unaggregated.tsv', sep='\\t', index=False)
EOF
    """
}

//
// 6) Draw contig‐level SVGs + PNGs under results/drawings/contigs_drawings
//
process DRAW_CONTIG_SVGS {
  tag "draw_contig_svgs"
  publishDir "./${params.outdir}/drawings/contigs_drawings", mode: 'copy'

  input:
    path top_contigs_tsv  // exon_pairs_scores_top_contigs.tsv

  output:
    path 'failures.log',      emit: contig_failures
    path 'kts_scripts',       emit: contig_kts
    path 'individual_svgs',   emit: contig_individual
    path 'pairs_drawings',    emit: contig_pairs

  script:
  """
  # Pre-create expected output dirs; create an empty failures.log file
  mkdir -p kts_scripts individual_svgs pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
  touch failures.log

  # Massage TSV into the format draw_pairs.py expects
  python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('${top_contigs_tsv}', sep='\\t')
df = df.rename(columns={
  'contig_start_1':'window_start_1',
  'contig_end_1':'window_end_1',
  'contig_start_2':'window_start_2',
  'contig_end_2':'window_end_2'
})
df.to_csv('to_draw_contigs.tsv', sep='\\t', index=False)
EOF

  # Generate the drawings
  python3 /app/draw_pairs.py \
    --tsv to_draw_contigs.tsv \
    --outdir . \
    --width 500 --height 500 \
    --highlight-colour "#00FF99" \
    --num-workers ${params.num_workers}

  # Organize pair SVGs/PNGs
  mkdir -p pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
  mv pair_*.svg pairs_drawings/pairs_svgs/ 2>/dev/null || true
  mv pair_*.png pairs_drawings/pairs_pngs/ 2>/dev/null || true
  """
}

//
// 7) Draw unaggregated‐window SVGs + PNGs under results/drawings/unagg_windows_drawings
//
process DRAW_UNAGG_SVGS {
  tag "draw_window_svgs"
  publishDir "./${params.outdir}/drawings/unagg_windows_drawings", mode: 'copy'

  input:
    path top_windows_tsv  // exon_pairs_scores_top_contigs.unaggregated.tsv

  output:
    path 'failures.log',      emit: window_failures
    path 'kts_scripts',       emit: window_kts
    path 'individual_svgs',   emit: window_individual
    path 'pairs_drawings',    emit: window_pairs

  script:
  """
  # Pre-create expected output dirs; create an empty failures.log file
  mkdir -p kts_scripts individual_svgs pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
  touch failures.log

  # We just feed the unaggregated TSV directly.
  cp ${top_windows_tsv} to_draw_windows.tsv
  
  # Generate the drawings
  python3 /app/draw_pairs.py \
    --tsv to_draw_windows.tsv \
    --outdir . \
    --width 500 --height 500 \
    --highlight-colour "#00FF99" \
    --num-workers ${params.num_workers}

  # Organize pair SVGs/PNGs
  mkdir -p pairs_drawings/pairs_svgs pairs_drawings/pairs_pngs
  mv pair_*.svg pairs_drawings/pairs_svgs/ 2>/dev/null || true
  mv pair_*.png pairs_drawings/pairs_pngs/ 2>/dev/null || true
  """
}


//
// 8a) Generate the contig‐level HTML report
//
process GENERATE_AGGREGATED_REPORT {
    tag "gen_agg_report"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path top_contigs_tsv     // exon_pairs_scores_top_contigs.tsv
      path contig_svgs         // drawings/contigs_drawings/individual_svgs

    output:
      path "exon_pairs_contigs_report.html"

    script:
    """
    python3 /app/generate_report.py \
      --pairs ${top_contigs_tsv} \
      --svg-dir ${contig_svgs} \
      --output exon_pairs_contigs_report.html
    """
}


//
// 8b) Generate the unaggregated‐window HTML report
//
process GENERATE_UNAGGREGATED_REPORT {
    tag "gen_unagg_report"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path top_windows_tsv     // exon_pairs_scores_top_contigs.unaggregated.tsv
      path window_svgs         // drawings/unagg_windows_drawings/individual_svgs

    output:
      path "exon_pairs_contigs_report.unaggregated.html"

    script:
    """
    python3 /app/generate_report.py \
      --pairs ${top_windows_tsv} \
      --svg-dir ${window_svgs} \
      --output exon_pairs_contigs_report.unaggregated.html
    """
}