import os.path as osp
from vlmeval.dataset.image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd
import re
import os
import string
import warnings

# Adapt ImageMCQDataset to load a local TSV and evaluate MCQs
class AfrimedQA(ImageMCQDataset):

    DATASET_URL = {"AfriMedQA": ""}
    DATASET_MD5 = {"AfriMedQA": ""}

    @classmethod
    def supported_datasets(cls):
        return ['AfrimedQA']

    def load_data(self, dataset="AfriMedQA", **kwargs):
        # Expect a local TSV at <LMUDataRoot()>/<dataset>.tsv
        data_path = osp.join(LMUDataRoot(), f"{dataset}.tsv")
        if not osp.exists(data_path):
            raise FileNotFoundError(f"Dataset file {data_path} does not exist.")
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            report_acc, report_acc_MMT, report_acc_MMSci, mcq_circular_eval, mcq_vanilla_eval
        )

        # Normalize dataset aliases used elsewhere
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]

        nproc = judge_kwargs.pop('nproc', 4)
        circular = False

        # Circular protocols for certain datasets
        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        # Judge selection: exact matching or LLM judge
        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, falling back to exact matching.')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set, falling back to exact matching.')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        # Load and normalize evaluation file
        data = load(eval_file)
        data = data.sort_values(by='index')

        # Keep MCQs if dataset includes question_type
        if 'split' in data.columns and 'question_type' in data.columns:
            data = data[data['question_type'] == 'MCQ']

        # Ensure predictions are strings
        data['prediction'] = [str(x) for x in data['prediction']]

        # If ground truth is in 'correct_option', mirror it to 'answer'
        if 'correct_option' in data.columns:
            data['answer'] = data['correct_option']

        def extract_choice(text):
            # Extract one of A-E from prediction text
            match = re.search(r'\b([A-E])\b', str(text).upper())
            if match:
                return match.group(1)
            return "INVALID"

        data['prediction'] = data['prediction'].apply(extract_choice)

        if 'index' not in data.columns:
            data.reset_index(inplace=True)

        cols = ['index', 'question', 'prediction', 'answer']
        cols = [c for c in cols if c in data.columns]

        print("\nSample Predictions vs. Ground Truth:\n")
        try:
            print(data[cols].head(10).to_markdown(index=False))
        except Exception:
            print(data[cols].head(10).to_string(index=False))

        # Filter to valid letter predictions
        data = data[data['prediction'].isin(['A', 'B', 'C', 'D', 'E'])]

        # Lowercase non-uppercase-single-letter keys for consistency
        for k in list(data.keys()):
            new_k = k if k in list(string.ascii_uppercase) else k.lower()
            if new_k != k:
                data[new_k] = data.pop(k)

        meta = self.data

        # Sanity check on metadata correctness column
        if 'correct_option' in meta.columns:
            num_missing_correct = meta['correct_option'].isna().sum()
            print(f"Number of questions missing correct_option: {num_missing_correct}")
        else:
            print("Column 'correct_option' not found in the dataset.")

        mcq_num = (meta['question_type'] == 'MCQ').sum() if 'question_type' in meta.columns else len(meta)
        print(f"Total number of MCQ questions: {mcq_num}")

        # Ensure eval file is aligned with meta indices
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])} if 'index' in meta and 'question' in meta else {}
        data_map = {x: y for x, y in zip(data['index'], data['question'])} if 'index' in data and 'question' in data else {}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        # Run evaluation protocol
        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        print("Eval file used for scoring:", eval_file)

        # Persist raw eval results
        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)

        # Compute hits
        if 'answer' in data.columns and 'prediction' in data.columns:
            data['hit'] = [
                int(str(pred).strip().upper() == str(ans).strip().upper())
                for pred, ans in zip(data['prediction'], data['answer'])
            ]

        # Choose accuracy reporter
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        else:
            acc = report_acc(data)

        # Save per-split accuracy table
        score_file = os.path.splitext(eval_file)[0] + '_acc_all.csv'
        print("Eval file:", eval_file)
        print("Suffix:", suffix)
        dump(acc, score_file)
        print("Score file to be written:", score_file)

        acc_map = {}

        # compare circular vs. vanilla accuracies for circular datasets
        if circular and os.environ.get('PRINT_VANILLA', None) == '1':
            acc_map['circular'] = acc

            # Vanilla Circ0 Acc
            data0_all = load(eval_file)
            data0_all['index'] = [int(x) for x in data0_all['index']]
            if 'g_index' in data0_all:
                data0_all['g_index'] = [int(x) for x in data0_all['g_index']]
                circ0 = data0_all[data0_all['g_index'] == data0_all['index']]
            else:
                offset = 1e6
                circ0 = data0_all[data0_all['index'] <= offset]

            result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_vanilla_result.pkl')
            data0 = mcq_vanilla_eval(model, circ0, meta, nproc, result_file, self.dataset_name)
            dump(data0, eval_file.replace(f'.{suffix}', f'_{name_str}_vanilla_circ0_result.{suffix}'))
            datac0 = load(eval_file.replace(f'.{suffix}', f'_{name_str}_vanilla_circ0_result.{suffix}'))
            acc_map['vanilla_0'] = report_acc(datac0)

            # Vanilla ALL Acc
            data_all = load(eval_file)
            dataall = mcq_vanilla_eval(model, data_all, meta, nproc, result_file, self.dataset_name)
            dump(dataall, eval_file.replace(f'.{suffix}', f'_{name_str}_vanilla_all_result.{suffix}'))
            data_va = load(eval_file.replace(f'.{suffix}', f'_{name_str}_vanilla_all_result.{suffix}'))
            acc_map['vanilla_all'] = report_acc(data_va)

            # Merge and label splits
            for k, v in acc_map.items():
                if 'split' not in v:
                    v['split'] = [None] * len(v)
                if len(v) == 1 and pd.isna(v['split'][0]):
                    v['split'] = [k]
                else:
                    assert not pd.isna(v['split'][0])
                    v['split'] = [k + '_' + sp for sp in v['split']]

            score_all = pd.concat([acc_map['vanilla_0'], acc_map['vanilla_all'], acc_map['circular']])

        else:
            if 'split' not in acc.columns:
                acc['split'] = ['test'] * len(acc)
            acc_map['main'] = acc
            score_all = acc

        # Append totals to each accuracy table
        total_questions = len(data)
        total_correct = data['hit'].sum() if 'hit' in data.columns else 0
        for acc_df in acc_map.values():
            acc_df['Total_Correct'] = int(total_correct)
            acc_df['Total_Questions'] = int(total_questions)

        # Save detailed per-question results
        full_data_file = eval_file.replace(f'.{suffix}', '_full_data.csv')
        data.to_csv(full_data_file, index=False, encoding='utf-8')
        print(f"Full data written to: {full_data_file}")

        print("================= Score All ==============")
        print(score_all)
        print("================= Score All ==============")
        score_file = eval_file.replace(f'.{suffix}', '_acc_all.csv')
        print("Score file to be written:", score_file)

        dump(score_all, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn(
                'AesBench VAL is a toy subset of AesBench TEST. For full results, use AesBench TEST.'
            )
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn(
                'Results differ from the original VisOnlyQA due to split changes and different prompts.'
            )

        return acc
