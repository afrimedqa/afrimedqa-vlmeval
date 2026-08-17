import os
import os.path as osp
import pandas as pd
import re
import string
import warnings

from vlmeval.dataset.text_mcq import TextMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

class AfrimedTextQA(TextMCQDataset):
    
    DATASET_URL = {"AfrimedTextQA": ""}
    DATASET_MD5 = {"AfrimedTextQA": ""}

    @classmethod
    def supported_datasets(cls):
        return ['AfrimedTextQA']

    def __init__(self, dataset="AfrimedTextQA", use_thinking_tag=True, data_dir=None, data_file=None, **kwargs):
        self.data_dir = data_dir
        self.data_file = data_file
        self.use_thinking_tag = use_thinking_tag
        super().__init__(dataset=dataset, data_dir=data_dir, data_file=data_file, **kwargs)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs = super().build_prompt(line)

        target_language = line.get('language', 'English')
        if isinstance(target_language, str):
            target_language = target_language.capitalize()

        if self.use_thinking_tag:
            cot_clinical_constraints = (
                "\n\nAs an expert clinician, select the correct option from the list of multiple choices for the clinical question. "
                "First, use the <thinking> tag to reason through the case step-by-step and rule out incorrect options. "
                "Then, provide a concise, high-yield clinical summary of your rationale in the <answer_reason> tag, as it will be reviewed by other physicians. "
                "Finally, provide the single letter of the correct option in the <final_answer> tag.\n\n"
                f"IMPORTANT: The clinical summary of rationale (<answer_reason>) MUST be written entirely in {target_language}. "
                "Do NOT include any medical disclaimers or AI caveats.\n\n"
                "Strictly format your output using the following XML tags in this exact order:\n"
                "<thinking>\n"
                "Your internal step-by-step differential diagnosis and distractor elimination here (language does not matter).\n"
                "</thinking>\n"
                "<answer_reason>\n"
                "Your concise, expert-level clinical summary here.\n"
                "</answer_reason>\n"
                "<final_answer>\n"
                "ONLY the single letter of the correct option (e.g., A, B, C, or D).\n"
                "</final_answer>"
            )
        else:
            cot_clinical_constraints = (
                "\n\nAs an expert clinician, select the correct option from the list of multiple choices for the clinical question. "
                "Provide a concise, high-yield clinical summary of your rationale in the <answer_reason> tag, as it will be reviewed by other physicians. "
                "Then, provide the single letter of the correct option in the <final_answer> tag.\n\n"
                f"IMPORTANT: The clinical summary of rationale (<answer_reason>) MUST be written entirely in {target_language}. "
                "Do NOT include any medical disclaimers or AI caveats.\n\n"
                "Strictly format your output using the following XML tags in this exact order:\n"
                "<answer_reason>\n"
                "Your concise, expert-level clinical summary here.\n"
                "</answer_reason>\n"
                "<final_answer>\n"
                "ONLY the single letter of the correct option (e.g., A, B, C, or D).\n"
                "</final_answer>"
            )

        for msg in msgs:
            if msg['type'] == 'text':
                msg['value'] += cot_clinical_constraints
                break

        return msgs


class AfrimedTextQA_Direct(AfrimedTextQA):
    @classmethod
    def supported_datasets(cls):
        return ['AfrimedTextQA_Direct']

    def __init__(self, dataset="AfrimedTextQA_Direct", use_thinking_tag=False, data_dir=None, data_file=None, **kwargs):
        super().__init__(dataset=dataset, use_thinking_tag=use_thinking_tag, data_dir=data_dir, data_file=data_file, **kwargs)

    def load_data(self, dataset="AfrimedTextQA", **kwargs):
        data_dir = kwargs.get('data_dir', None)
        data_file = kwargs.get('data_file', None)

        if data_file and osp.exists(data_file):
            data_path = data_file
        elif data_dir and osp.exists(osp.join(data_dir, f"{dataset}.tsv")):
            data_path = osp.join(data_dir, f"{dataset}.tsv")
        elif osp.exists(dataset):
            data_path = dataset
        elif osp.exists(f"{dataset}.tsv"):
            data_path = f"{dataset}.tsv"
        elif (hasattr(self.__class__, "DATASET_URL")
            and dataset in self.__class__.DATASET_URL
            and osp.exists(self.__class__.DATASET_URL[dataset])):
            data_path = self.__class__.DATASET_URL[dataset]
        else:
            data_path = osp.join(LMUDataRoot(), f"{dataset}.tsv")

        if not osp.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        data = load(data_path)
        if 'question_type' in data.columns:
            data = data[data['question_type'] == 'MCQ'].reset_index(drop=True)
        return data
    


    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        logger.info("Starting evaluation for Afrimed Text-Only MCQA...")

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

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        # Align ground truth metadata
        meta = self.data
        if 'question_type' in meta.columns:
            meta = meta[meta['question_type'] == 'MCQ'].copy()

        # Handle answer column — prefer correct_option in eval file, fall back to meta
        if 'correct_option' in data.columns:
            data['answer'] = data['correct_option']
        elif 'answer' not in data.columns and 'correct_option' in meta.columns:
            data = data.merge(meta[['index', 'correct_option']], on='index', how='left')
            data['answer'] = data['correct_option']

        if 'index' not in data.columns:
            data.reset_index(inplace=True)

        def extract_thinking(text: str) -> str:
            text = str(text).strip()
            m = re.search(r'<thinking>\s*(.*?)\s*</thinking>', text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def extract_rationale(text: str) -> str:
            text = str(text).strip()
            # 1. Match XML tag <answer_reason>...</answer_reason>
            m_xml = re.search(r'<answer_reason>\s*(.*?)\s*</answer_reason>', text, re.DOTALL | re.IGNORECASE)
            if m_xml:
                return m_xml.group(1).strip()
            # 2. If explicit ANSWER REASON tag exists
            m_reason = re.search(r'(?i)answer\s+reason\s*:\s*(.*?)(?:final\s+answer|answer\s*:|$)', text, re.DOTALL)
            if m_reason:
                return m_reason.group(1).strip()
            # 3. Find where the answer line starts and take everything before it as the rationale
            m = re.search(r'(?i)(final\s+answer|answer\s*is|answer)\s*:?\s*\*?\*?[A-E]', text)
            if m:
                rationale = text[:m.start()].strip()
                rationale = re.sub(r'^(?i)(rationale|answer\s+reason)\s*:\s*', '', rationale).strip()
                return rationale
            # Fallback: if no answer marker is found, return the whole text as rationale
            rationale = re.sub(r'^(?i)(rationale|answer\s+reason)\s*:\s*', '', text).strip()
            return rationale

        def extract_choice(text):
            text = str(text).strip()

            # 1. Match inside XML tag <final_answer>...</final_answer>
            m_xml = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', text, re.DOTALL | re.IGNORECASE)
            if m_xml:
                inner = m_xml.group(1).strip()
                m_letter = re.search(r'\b([A-E])\b', inner, re.IGNORECASE)
                if m_letter:
                    return m_letter.group(1).upper()

            # 2. Tag without closing or formatted inline
            match = re.search(r'<final_answer>\s*\*?\*?([A-E])\b', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            match = re.search(r'FINAL ANSWER:\s*\*?\*?([A-E])', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            match = re.search(r'\*\*(A|B|C|D|E)(?:\.|\*\*)', text)
            if match: return match.group(1)

            match = re.search(r'answer is (A|B|C|D|E)', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            match = re.search(r'\b(A|B|C|D|E)\.', text)
            if match: return match.group(1)

            if text.upper() in ['A', 'B', 'C', 'D', 'E']:
                return text.upper()

            match = re.match(r'^([A-E])\b', text, re.IGNORECASE)
            if match: return match.group(1).upper()

            return "INVALID"

        data['parsed_thinking'] = data['prediction'].apply(extract_thinking)
        data['parsed_reason'] = data['prediction'].apply(extract_rationale)
        data['prediction'] = data['prediction'].apply(extract_choice)

        cols = ['index', 'question', 'prediction', 'answer']
        cols = [c for c in cols if c in data.columns]

        print("\nSample Predictions vs. Ground Truth:\n")
        try:
            print(data[cols].head(10).to_markdown(index=False))
        except Exception:
            print(data[cols].head(10).to_string(index=False))

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

        print(f"Total number of MCQ questions: {len(meta)}")

        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])} if 'index' in meta and 'question' in meta else {}
        data_map = {x: y for x, y in zip(data['index'], data['question'])} if 'index' in data and 'question' in data else {}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

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