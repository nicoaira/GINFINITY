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

    // 4) Aggregate + enrich in one step
    def (enriched_all_ch, enriched_unagg_ch) = AGGREGATE_METRIC(sorted_distances_ch, input_tsv_ch)

    // 5) Legacy: also filter top-N distances for pair-drawing & report
    def topn_ch = FILTER_TOP_N(sorted_distances_ch)
    def svg_ch  = DRAW_WINDOWS_PAIRS(topn_ch)
    GENERATE_HTML_REPORT(topn_ch, svg_ch)
}


process GENERATE_EMBEDDINGS {
    tag "generate_embeddings"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path input_file

    output:
      path "${params.outdir}/embeddings.tsv", emit: embeddings

    script:
    """
    python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('${input_file}', sep='\\t')
df['seq_len'] = df['exon_sequence'].str.len()
df.to_csv('with_seq_len.tsv', sep='\\t', index=False)
EOF

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
      path "exon_pairs_scores_all_contigs.tsv",               emit: enriched_all
      path "exon_pairs_scores_all_contigs.unaggregated.tsv",  emit: enriched_unagg

    script:
    """
    # 1) Run aggregated_metric.py to produce raw intermediates
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

    # 2) Enrich those intermediates with gene_name, sequence, structure
    python3 - << 'EOF'
import pandas as pd

# Load the raw contig tables
df_all = pd.read_csv('raw_contigs.tsv', sep='\\t')
df_un  = pd.read_csv('raw_contigs.unaggregated.tsv', sep='\\t')

# Load the original input for metadata
df_meta = pd.read_csv('${input_tsv}', sep='\\t', dtype=str)
idcol     = 'exon_id'
structcol = '${params.structure_column_name}'

# Prepare side-1 metadata
m1 = df_meta[[idcol,'gene_name','exon_sequence', structcol]]\
      .rename(columns={
         idcol: 'exon_id_1',
         'gene_name': 'gene_name_1',
         'exon_sequence': 'sequence_1',
         structcol: 'secondary_structure_1'
      })

# Prepare side-2 metadata
m2 = df_meta[[idcol,'gene_name','exon_sequence', structcol]]\
      .rename(columns={
         idcol: 'exon_id_2',
         'gene_name': 'gene_name_2',
         'exon_sequence': 'sequence_2',
         structcol: 'secondary_structure_2'
      })

# Join & write enriched contigs
df_all = df_all.merge(m1, on='exon_id_1', how='left')\
               .merge(m2, on='exon_id_2', how='left')
df_all.to_csv('exon_pairs_scores_all_contigs.tsv', sep='\\t', index=False)

# Join & write enriched unaggregated windows
df_un = df_un.merge(m1, on='exon_id_1', how='left')\
             .merge(m2, on='exon_id_2', how='left')
df_un.to_csv('exon_pairs_scores_all_contigs.unaggregated.tsv', sep='\\t', index=False)
EOF
    """
}


process FILTER_TOP_N {
    tag "filter_top_n"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path distances

    output:
      path "top_${params.top_n}.tsv", emit: topn

    script:
    """
    python3 - << 'EOF'
import pandas as pd
df = pd.read_csv('${distances}', sep='\\t')
df_sorted = df.sort_values('distance').head(${params.top_n})
df_sorted.to_csv('top_${params.top_n}.tsv', sep='\\t', index=False)
EOF
    """
}


process DRAW_WINDOWS_PAIRS {
    tag "draw_windows_pairs"
    publishDir "./${params.outdir}/structures_plots", mode: 'copy'

    input:
      path top_n_file

    output:
      path "individual_svgs", emit: svgs

    when:
      params.draw_pairs

    script:
    """
    python3 /app/draw_pairs.py \
      --tsv ${top_n_file} \
      --width 500 \
      --height 500 \
      --highlight-colour "#00FF99" \
      --num-workers ${params.num_workers} \
      --outdir .
    """
}


process GENERATE_HTML_REPORT {
    tag "html_report"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
      path top_n_file
      path svg_dir

    output:
      path "report.html"

    when:
      params.draw_pairs

    script:
    """
    python3 /app/generate_report.py \
      --pairs ${top_n_file} \
      --svg-dir ${svg_dir} \
      --output report.html
    """
}
