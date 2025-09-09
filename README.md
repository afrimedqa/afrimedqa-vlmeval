# AfriMedQA with VLMEvalKit (Quick Start)

The AfriMedQA dataset class extends VLMEvalKit's `ImageMCQDataset` to load a local TSV of questions and
evaluate with **exact matching** or an **LLM judge**. Evaluation support for SAQ is not included (**yet to decide on appropriate evaluation metrics**) 

------------------------------------------------------------------------

## Quick Start

### 1) Create New Virtual Environment with Conda and Install package requirements

``` bash
# Create a new conda env with Python 3.10

conda create -n afrimedqa_vlmeval python=3.10 -y
conda activate afrimedqa_vlmeval

# Install packages
pip install -r requirements.txt
# If converting HTML → TSV (e.g. using html_to_tsv.py), install the following packages:
```

### 2) Prepare the dataset

Ensure you have the CSV/TSV of the language you want to evaluate on. The CSV/TSV must have the following columns:
- `image_path`
- `question`
- `A`
- `B`
- `C`
- `D`
- `answer`



#### (Optional) HTML → TSV Conversion

If dataset is in **HTML** form, convert it to TSV with the script below:

``` bash
python html_to_tsv.py --html All_Pics_Questions/All_Pics_Questions.html --out AfrimedQA_en.tsv
```

**Defaults** (omit flags if these match your layout):
- `--html All_Pics_Questions/All_Pics_Questions.html`
- `--out AfrimedQA_en.tsv`

**About the script:**
The script converts the AfriMedQA dataset from its original HTML format into a clean TSV file, extracting questions, options, and correct answers while embedding any images in base64 format for evaluation with VLMEvalKit.


### Accessing Evaluation Images

The images have been placed in a zip file on Google Drive. Please download the zip file (see below) and unzip each image into `/lmudata/images/afrimedqa` or any other path of your choice.

#### How to Download the Image Zip File

Please reach out for the full ID of the image on Google Drive. The ID below has been anonymized for privacy.

```bash
pip install gdown
gdown 1NotzG0VfRdQSZS2kd64Lq****
```


### 3) Set up LMUData Directory

Export the dataset path and copy the TSV into place:

``` bash
export LMUData=/path/to/LMUData
lang=en
question_type=MCQ
model=Gemma3-12B
img_path=images/afrimedqa
```

> **Note:** VLMEvalKit uses `LMUDataRoot()` to locate datasets. By setting `LMUData`, you tell it where to look.

### 4) Model Selection

Please see `vlmeval/config.py` for all the available open source models you can choose from. You are also welcome to extend the list (like we did for MedGemma) with a PR if you can.

### 5) Run an Evaluation

``` bash
bash scripts/run.sh
```

**Script parameters:**
- `--data AfrimedQA` → selects the benchmark
- `--lang` → indicates the language you are working with (we use "en" for English in this example)
- `--question_type` → the question type: MCQ or SAQ
- `--img_path` → the path where you saved the images
- `--model Idefics3-8B-Llama3` → chooses the LVLM
- `--work-dir` → directory where logs/results are saved

### 6) Outputs

- Per-question predictions and hits
- Accuracy summary CSV: `_acc_all.csv`
- Full per-item results: `_full_data.csv`

------------------------------------------------------------------------

## About the `AfrimedQA` Class

- **Loads local TSV** from `$LMUData/AfrimedQA_en.tsv`
- **Ensures model predictions** are converted into one of the valid answer choices (A, B, C, D, or E)
- **Judging modes**
  - Exact matching (default)
  - LLM judge if an OpenAI key is set
- **Reports** test and validation accuracy and writes results to CSV file

------------------------------------------------------------------------

## Notes on Benchmarks, Models, and Versions

- [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [**Download All DETAILED Results**](http://opencompass.openxlab.space/assets/OpenVLM.json)
- Check **Supported Benchmarks**: [VLMEvalKit Features -- Benchmarks (70+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)
- Check **Supported LMMs**: [VLMEvalKit Features -- LMMs (200+)](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)

### Transformers Version Recommendations

Some models require specific `transformers` versions:

- `transformers==4.33.0` → Qwen, Monkey, InternLM-XComposer, mPLUG-Owl2, OpenFlamingo v2, IDEFICS, VisualGLM, etc.
- `transformers==4.36.2` → Moondream1
- `transformers==4.37.0` → LLaVA, ShareGPT4V, CogVLM, EMU2, Yi-VL, DeepSeek-VL, InternVL, etc.
- `transformers==4.40.0` → IDEFICS2, Bunny-Llama3, MiniCPM-Llama3-V2.5, Phi-3-Vision, etc.
- `transformers==4.42.0` → AKI
- `transformers==4.44.0` → Moondream2, H2OVL
- `transformers==4.45.0` → Aria
- `transformers==latest` → LLaVA-Next, PaliGemma-3B, Chameleon, Ovis, Mantis, Idefics-3, GLM-4v-9B, etc.

### Torchvision Version

- Use `torchvision>=0.16` for Moondream series, Aria

### Flash-attn Version

- Use `pip install flash-attn --no-build-isolation` for Aria

### Demo

``` python
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images?'])
print(ret)  # There are two apples in the provided images.
```


