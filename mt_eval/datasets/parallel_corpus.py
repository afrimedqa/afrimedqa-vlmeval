import pandas as pd


def load_parallel_corpus(
    source_file: str,
    target_file: str,
) -> pd.DataFrame:
    """Load English and target-language TSVs and merge on sample_id to produce
    aligned translation pairs. All question types (MCQ and SAQ) are included.

    Returns a DataFrame with columns:
        sample_id, question_src, answer_src, question_tgt, answer_tgt,
        specialty, country, gender, question_type
    """
    # Skip the base64 image column — MT eval is text-only
    src = pd.read_csv(source_file, sep='\t', usecols=lambda c: c != 'image')
    tgt = pd.read_csv(target_file, sep='\t', usecols=lambda c: c != 'image')

    if src.empty:
        raise ValueError(f"No rows found in source file: {source_file}")
    if tgt.empty:
        raise ValueError(f"No rows found in target file: {target_file}")

    keep_src = ['sample_id', 'question', 'answer', 'specialty', 'country', 'gender', 'question_type']
    keep_tgt = ['sample_id', 'question', 'answer']

    # Only keep columns that actually exist
    keep_src = [c for c in keep_src if c in src.columns]
    keep_tgt = [c for c in keep_tgt if c in tgt.columns]

    merged = src[keep_src].merge(
        tgt[keep_tgt],
        on='sample_id',
        suffixes=('_src', '_tgt'),
    )

    n_src = len(src)
    n_tgt = len(tgt)
    n_merged = len(merged)
    print(f"Translation pairs: {n_src} source / {n_tgt} target → {n_merged} aligned")

    if 'question_type' in merged.columns:
        for qt, count in merged['question_type'].value_counts().items():
            print(f"  {qt}: {count}")

    unmatched = set(src['sample_id']) - set(tgt['sample_id'])
    if unmatched:
        print(f"Warning: {len(unmatched)} source sample_ids have no target match: {sorted(unmatched)}")

    return merged
