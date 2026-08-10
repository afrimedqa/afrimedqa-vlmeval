import os.path as osp
from vlmeval.dataset.image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd
import re
import os
import string
import warnings
import re

# load and evaluate MCQs
class AfrimedQA(ImageMCQDataset):

    DATASET_URL = {"AfriMedQA": ""}
    DATASET_MD5 = {"AfriMedQA": ""}

    @classmethod
    def supported_datasets(cls):
        return ['AfrimedQA']

    def load_data(self, dataset="AfrimedQA", **kwargs):
        if (hasattr(self.__class__, "DATASET_URL")
            and dataset in self.__class__.DATASET_URL
            and osp.exists(self.__class__.DATASET_URL[dataset])):
            data_path = self.__class__.DATASET_URL[dataset]
        else:
            data_path = osp.join(LMUDataRoot(), f"{dataset}.tsv")

        if not osp.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = load(data_path)

        # Filter for MCQ ONLY
        if 'question_type' in df.columns:
            df = df[df['question_type'] == 'MCQ'].copy()

        return df
    

    def build_prompt(self, line):
        # Get the default framework prompt
        msgs = super().build_prompt(line)
        
        target_language = line.get('language', 'English')
        if isinstance(target_language, str):
            target_language = target_language.capitalize()

        cot_clinical_constraints = (
            f"\n\nIn a single short paragraph (3-4 sentences), briefly explain your clinical reasoning entirely in {target_language}. "
            "Then state your final answer on a new line in this exact format:\n"
            "Answer: <letter>\n"
            "where <letter> is exactly one of A, B, C, or D. Do not add anything after the answer line."
        )

        for msg in msgs:
            if msg['type'] == 'text':
                msg['value'] += cot_clinical_constraints
                break

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        logger.info("Starting evaluation for Afrimed MCQA...")
        from .utils.multiple_choice import (
            report_acc, report_acc_MMT, report_acc_MMSci, mcq_circular_eval, mcq_vanilla_eval
        )

        dataset = self.dataset_name
        nproc = judge_kwargs.pop('nproc', 4)
        circular = False

        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

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

        # Load data
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        # Align Ground Truth Metadata
        meta = self.data
        if 'question_type' in meta.columns:
            meta = meta[meta['question_type'] == 'MCQ'].copy()

        # Handle Answer Mirroring
        if 'correct_option' in data.columns:
            data['answer'] = data['correct_option']
        elif 'answer' not in data.columns and 'correct_option' in meta.columns:
            data = data.merge(meta[['index', 'correct_option']], on='index', how='left')
            data['answer'] = data['correct_option']

        if 'index' not in data.columns:
            data.reset_index(inplace=True)

        def extract_rationale(text: str) -> str:
            text = str(text).strip()
            # Find where the answer line starts and take everything before it as the rationale
            m = re.search(r'(?i)(final\s+answer|answer\s*is|answer)\s*:?\s*\*?\*?[A-E]', text)
            if m:
                rationale = text[:m.start()].strip()
                # Strip an optional leading "Rationale:" label the model may add
                rationale = re.sub(r'^(?i)rationale\s*:\s*', '', rationale).strip()
                return rationale
            # Fallback: if no answer marker is found, return the whole text as rationale
            rationale = re.sub(r'^(?i)rationale\s*:\s*', '', text).strip()
            return rationale

        def extract_choice(text):

            text = str(text).strip()


            match = re.search(r'FINAL ANSWER:\s*\*?\*?([A-E])', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            match = re.search(r'answer is:?\s*\*?\*?([A-E])', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            match = re.search(r'[Aa]nswer:\s*\*?\*?([A-E])', text)
            if match: return match.group(1).upper()


            match = re.search(r'\*\*(A|B|C|D|E)(?:\.|\*\*)', text)
            if match: return match.group(1)

            match = re.search(r'\b(A|B|C|D|E)\.', text)
            if match: return match.group(1)


            if text.upper() in ['A', 'B', 'C', 'D', 'E']:
                return text.upper()

            match = re.match(r'^([A-E])\b', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            # Last resort: find the last standalone A-E in the text
            match = re.search(r'\b([A-E])\b(?=[^A-E]*$)', text)
            if match: return match.group(1).upper()

            return "INVALID"

        # Extract rationale before overwriting prediction with the letter
        data['answer_rationale'] = data['prediction'].apply(extract_rationale)

        # Apply our custom regex to clean the predictions
        data['prediction'] = data['prediction'].apply(extract_choice)

        # Lowercase keys for consistency
        for k in list(data.keys()):
            new_k = k if k in list(string.ascii_uppercase) else k.lower()
            if new_k != k:
                data[new_k] = data.pop(k)

        if 'correct_option' in meta.columns:
            num_missing_correct = meta['correct_option'].isna().sum()
            print(f"Number of questions missing correct_option: {num_missing_correct}")
        else:
            print("Column 'correct_option' not found in the dataset.")

        mcq_num = len(meta)
        print(f"Total number of MCQ questions: {mcq_num}")

        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])} if 'index' in meta and 'question' in meta else {}
        data_map = {x: y for x, y in zip(data['index'], data['question'])} if 'index' in data and 'question' in data else {}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        #  Extraction and LLM Judging
        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        print("Eval file used for scoring:", eval_file)

        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)


        if 'answer' in data.columns and 'prediction' in data.columns:
            data['hit'] = [
                int(str(pred).strip().upper() == str(ans).strip().upper())
                for pred, ans in zip(data['prediction'], data['answer'])
            ]


        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        else:
            acc = report_acc(data)

        score_file = os.path.splitext(eval_file)[0] + '_acc_all.csv'
        print("Eval file:", eval_file)
        print("Suffix:", suffix)
        dump(acc, score_file)
        print("Score file to be written:", score_file)

        acc_map = {'main': acc}
        score_all = acc

        total_questions = len(data)
        total_correct = data['hit'].sum() if 'hit' in data.columns else 0
        for acc_df in acc_map.values():
            acc_df['Total_Correct'] = int(total_correct)
            acc_df['Total_Questions'] = int(total_questions)

        full_data_file = eval_file.replace(f'.{suffix}', '_full_data.csv')
        data.to_csv(full_data_file, index=False, encoding='utf-8')
        print(f"Full data written to: {full_data_file}")

        print("================= Score All ==============")
        print(score_all)
        print("================= Score All ==============")
        score_file = eval_file.replace(f'.{suffix}', '_acc_all.csv')
        print("Score file to be written:", score_file)

        dump(score_all, score_file)

        return acc