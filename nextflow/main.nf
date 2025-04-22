#!/usr/bin/env nextflow
// Nextflow pipeline to generate embeddings, compute distances, and filter top N smallest distances

workflow {
    // decide where to get embeddings
    def embeddings_ch = params.embeddings_file ?
        Channel.fromPath(params.embeddings_file) :
        GENERATE_EMBEDDINGS(Channel.fromPath(params.input))

    def distances_ch = COMPUTE_DISTANCES(embeddings_ch)
    def topn_ch = FILTER_TOP_N(distances_ch)
    DRAW_WINDOWS_PAIRS(topn_ch)
}

process GENERATE_EMBEDDINGS {
    tag "generate_embeddings"
    // publishDir "/home/nicolas/programs/GINFINITY/nextflow/results"
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
      ${params.L       ? "--L ${params.L}"                       : ''} \
      ${params.keep_paired_neighbors ? '--keep_paired_neighbors' : ''} \
      --retries ${params.retries}
    """
}

process COMPUTE_DISTANCES {
    tag "compute_distances"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
    path embeddings

    output:
    path "distances.tsv", emit: distances

    script:
    """
    python3 ${baseDir}/../compute_distances.py --input ${embeddings} \
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

process FILTER_TOP_N {
    tag "filter_top_n"
    publishDir "./${params.outdir}", mode: 'copy'

    input:
    path distances

    output:
    path "top_${params.top_n}.tsv"

    script:
    """
    python3 - << 'EOF'
import pandas as pd

df = pd.read_csv('${distances}', sep='\t')
df_sorted = df.sort_values('distance').head(${params.top_n})
df_sorted.to_csv('top_${params.top_n}.tsv', sep='\t', index=False)
EOF
    """
}

process DRAW_WINDOWS_PAIRS {
    tag "draw_windows_pairs"
    publishDir "./${params.outdir}/structures_plots", mode: 'copy'

    input:
    path top_n_file

    when:
    params.draw_pairs

    script:
    """
    python3 /app/draw_pairs.py \
      --tsv ${top_n_file} \
      --outdir ${params.outdir}/structures_plots
    """
}