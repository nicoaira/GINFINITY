#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * Nextflow pipeline: embeddings → distances → sort → top-N → draw → HTML report
 */
workflow {

    // — Embeddings (unchanged)
    def embeddings_ch = params.embeddings_file ?
        Channel.fromPath(params.embeddings_file) :
        GENERATE_EMBEDDINGS(Channel.fromPath(params.input))

    // — Compute raw distances (no longer published)
    def distances_ch = COMPUTE_DISTANCES(embeddings_ch)

    // — Sort distances → produces distances.sorted.tsv
    def sorted_distances_ch = SORT_DISTANCES(distances_ch)

    // — Filter Top-N on the sorted distances
    def topn_ch = FILTER_TOP_N(sorted_distances_ch)

    // — Draw the top-N pairs
    def svg_ch = DRAW_WINDOWS_PAIRS(topn_ch)

    // — Generate the HTML report
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
    python3 ${baseDir}/modules/predict_embedding.py \
      --input ${input_file} \
      --model_path ${params.model_path} \
      --output ${params.outdir}/embeddings.tsv \
      --structure_column_name ${params.structure_column_name} \
      ${params.structure_column_num != null ? "--structure_column_num ${params.structure_column_num}" : ''} \
      --header ${params.header} \
      --device ${params.device} \
      --num_workers ${params.num_workers} \
      ${params.subgraphs ? '--subgraphs' : ''} \
      ${params.L       ? "--L ${params.L}"         : ''} \
      ${params.keep_paired_neighbors ? '--keep_paired_neighbors' : ''} \
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
# Read raw distances
df = pd.read_csv('${distances}', sep='\\t')
# Sort ascending by distance
df.sort_values('distance', inplace=True)
# Write out the sorted TSV
df.to_csv('distances.sorted.tsv', sep='\\t', index=False)
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
