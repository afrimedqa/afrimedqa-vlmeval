"""
MT Evaluation entry point.

Usage (from project root):
    python -m mt_eval.run mt_eval/configs/en_twi_vllm.json
"""

import argparse
import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch

from mt_eval.datasets import load_parallel_corpus
from mt_eval.metrics import score_chrf, score_chrf_corpus, score_ssa_comet
from mt_eval.models import build_model
from mt_eval.models.seq2seq_model import Seq2SeqModel


SYSTEM_PROMPT = (
    "You are an expert technical translator specializing in clinical healthcare, "
    "specifically focusing on the African healthcare context. "
    "Your task is to translate the following Question and Answer pair from English into {target_lang}.\n\n"
    "Translation Constraints:\n\n"
    "Domain Accuracy: You must ensure all vocabulary, phrasing, and idioms strictly align with "
    "standard medical terminology relevant to the African healthcare context in {target_lang}. "
    "Avoid literal word-for-word translations if a more accurate, localized clinical term exists "
    "in {target_lang}.\n\n"
    "Logical Co-Dependence: The Question and the Answer are an inseparable pair. "
    "You must read both the Question and the Answer before translating either. "
    "Use the Answer to disambiguate any missing context or unclear medical terms in the Question.\n\n"
    "Strict Translation Binding: You are ONLY translating the provided text. "
    "Do not attempt to answer the Question yourself, solve the medical query, or generate new information. "
    "The translated Answer must be an exact semantic translation of the English Answer, "
    "preserving the original clinical meaning exactly as it is written.\n\n"
    "Output Format: Respond ONLY with the translated pair in this exact format:\n"
    "Question: <translated question>\n"
    "Answer: <translated answer>\n\n"
    "STRICT RULES — failure to follow these will make the output unusable:\n"
    "1. Output ONLY the two lines above. Do not add any explanation, reasoning, notes, commentary, "
    "or chain-of-thought before or after the translation.\n"
    "2. Do not refuse. This is a legitimate medical translation task. Always produce a translation.\n"
    "3. The Answer line must contain only the translated answer text — do not prepend words like "
    "'Diagnosis:' or 'Answer:' inside the answer itself.\n"
    "4. Translate everything into {target_lang}. Do not leave any part of the output in English "
    "unless it is a proper medical term with no established {target_lang} equivalent "
    "(e.g. a drug name or acronym)."
)


def build_messages(row: dict, target_lang: str) -> list[dict]:
    system = SYSTEM_PROMPT.format(target_lang=target_lang)
    user = f"Question: {row['question_src']}\nAnswer: {row['answer_src']}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_output(text: str) -> tuple[str, str]:
    """Extract translated question and answer from model output."""
    q_match = re.search(r'(?i)Question:\s*(.+?)(?=\nAnswer:|\Z)', text, re.DOTALL)
    a_match = re.search(r'(?i)Answer:\s*(.+?)$', text, re.DOTALL)
    hyp_q = q_match.group(1).strip() if q_match else ""
    hyp_a = a_match.group(1).strip() if a_match else ""

    # Fallback: if labels are missing but output has two non-empty lines, use line split
    if not hyp_q and not hyp_a:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) >= 2:
            hyp_q, hyp_a = lines[0], lines[-1]

    return hyp_q, hyp_a


def run(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    data_cfg = cfg['data']
    source_lang = data_cfg['source_lang']
    target_lang = data_cfg['target_lang']

    pairs = load_parallel_corpus(
        source_file=data_cfg['source_file'],
        target_file=data_cfg['target_file'],
    )

    model = build_model(cfg['model'])
    model_tag = cfg['model']['model_name'].replace('/', '_')

    output_dir = Path(cfg.get('output_dir', 'outputs/mt_eval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source_lang}_{target_lang}_{model_tag}"

    is_seq2seq = isinstance(model, Seq2SeqModel)

    results = []
    for _, row in pairs.iterrows():
        if is_seq2seq:
            hyp_q = model.translate_text(str(row['question_src']))
            hyp_a = model.translate_text(str(row['answer_src']))
            raw_output = f"Question: {hyp_q}\nAnswer: {hyp_a}"
        else:
            messages = build_messages(row, target_lang)
            raw_output = model.translate(messages)
            hyp_q, hyp_a = parse_output(raw_output)

        ref_q = str(row['question_tgt'])
        ref_a = str(row['answer_tgt'])

        results.append({
            'sample_id': row['sample_id'],
            'specialty': row.get('specialty', ''),
            'country': row.get('country', ''),
            'src_question': row['question_src'],
            'src_answer': row['answer_src'],
            'ref_question': ref_q,
            'ref_answer': ref_a,
            'hyp_question': hyp_q,
            'hyp_answer': hyp_a,
            'raw_output': raw_output,
            'chrf_question': score_chrf(hyp_q, ref_q),
            'chrf_answer': score_chrf(hyp_a, ref_a),
            'chrf_combined': score_chrf(f"{hyp_q} {hyp_a}", f"{ref_q} {ref_a}"),
        })
        print(f"  [{row['sample_id']}] chrF Q={results[-1]['chrf_question']:.1f} "
              f"A={results[-1]['chrf_answer']:.1f}")

    df = pd.DataFrame(results)

    # Free vLLM GPU memory before loading COMET
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # SSA-COMET (batch, uses src English as source)
    print("\nRunning SSA-COMET scoring...")
    try:
        q_sent, q_sys = score_ssa_comet(
            df['src_question'].tolist(),
            df['hyp_question'].tolist(),
            df['ref_question'].tolist(),
        )
        a_sent, a_sys = score_ssa_comet(
            df['src_answer'].tolist(),
            df['hyp_answer'].tolist(),
            df['ref_answer'].tolist(),
        )
        combined_sent, combined_sys = score_ssa_comet(
            (df['src_question'] + ' ' + df['src_answer']).tolist(),
            (df['hyp_question'] + ' ' + df['hyp_answer']).tolist(),
            (df['ref_question'] + ' ' + df['ref_answer']).tolist(),
        )
        df['ssa_comet_question'] = q_sent
        df['ssa_comet_answer'] = a_sent
        df['ssa_comet_combined'] = combined_sent
        ssa_comet_ok = True
    except Exception as e:
        print(f"  WARNING: SSA-COMET failed — {e}")
        q_sys = a_sys = combined_sys = None
        ssa_comet_ok = False

    results_file = output_dir / f"{stem}_results.csv"
    df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\nPer-question results saved to: {results_file}")

    # Corpus-level chrF (more stable than mean of sentence scores)
    corpus_chrf_q = score_chrf_corpus(
        df['hyp_question'].tolist(), df['ref_question'].tolist()
    )
    corpus_chrf_a = score_chrf_corpus(
        df['hyp_answer'].tolist(), df['ref_answer'].tolist()
    )
    corpus_chrf_combined = score_chrf_corpus(
        (df['hyp_question'] + ' ' + df['hyp_answer']).tolist(),
        (df['ref_question'] + ' ' + df['ref_answer']).tolist(),
    )

    summary_row = {
        'source_lang': source_lang,
        'target_lang': target_lang,
        'model': cfg['model']['model_name'],
        'n_pairs': len(df),
        # Sentence-level means
        'mean_chrf_question': round(df['chrf_question'].mean(), 2),
        'mean_chrf_answer': round(df['chrf_answer'].mean(), 2),
        'mean_chrf_combined': round(df['chrf_combined'].mean(), 2),
        # Corpus-level chrF
        'corpus_chrf_question': round(corpus_chrf_q, 2),
        'corpus_chrf_answer': round(corpus_chrf_a, 2),
        'corpus_chrf_combined': round(corpus_chrf_combined, 2),
    }

    if ssa_comet_ok:
        summary_row.update({
            'ssa_comet_question': round(q_sys, 4),
            'ssa_comet_answer': round(a_sys, 4),
            'ssa_comet_combined': round(combined_sys, 4),
        })

    summary = pd.DataFrame([summary_row])

    summary_file = output_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_file, index=False, encoding='utf-8')

    print("\n========== MT Evaluation Summary ==========")
    print(summary.T.to_string(header=False))
    print("===========================================")
    print(f"Summary saved to: {summary_file}")

    return df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MT evaluation for AfriMedQA SAQ pairs')
    parser.add_argument('config', help='Path to config JSON (e.g. mt_eval/configs/en_twi_vllm.json)')
    args = parser.parse_args()
    run(args.config)
