# MedFuse: Train and Test Guide

This guide shows how to set up the environment, prepare data, and run training/testing for MedFuse. It follows the structure of the original SCANE README, adapted to MedFuse-only workflows.

## 1) Environment

Recommended versions:
- Python 3.8
- PyTorch 1.13.1
- CUDA 11.x (if using GPU)

Setup with conda:
```bash
conda create --name MedFuse python=3.8
conda activate MedFuse
pip install -r requirements.txt
```

Or with venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Data

Datasets:
- PhysioNet2012 (P12): provided.
- MIMIC-III (MI3): license required (https://physionet.org/content/mimiciii/).
- HCC: protected by IRB.

Default data root
- By default, configs expect data under either:
  - ./data/P12, ./data/MI3, ... or

- Check and adjust dataset.data_path in your config files:
  - ./config/overall_result/P12/medfuse.json
  - ./config/overall_result/MI3/medfuse.json
  - ./config/overall_result/HCC/medfuse.json

```

## 4) Train and Test via helper script

Make the script executable:
```bash
chmod +x overall_result.sh
```

Train:
```bash
# P12
./overall_result.sh train P12
# MI3 (license required)
./overall_result.sh train MI3
# HCC (IRB required)
./overall_result.sh train HCC
```

Test with provided checkpoints:
```bash
# P12
./overall_result.sh test P12
# MI3 (license required)
./overall_result.sh test MI3
# HCC (IRB required)
./overall_result.sh test HCC
```

The script uses:
- Configs: ./config/overall_result/{P12|MI3|HCC}/medfuse.json
- Checkpoints (for test mode): ./exp_output/model_ckt_{P12|MI3|HCC}/medfuse/{config.json, ckt.pth}

## 5) Run main.py directly (optional)

Train:
```bash
python3 main.py \
  -e overall_result \
  -a MI3_MedFuse \
  -m train \
  -c ./config/overall_result/MI3/medfuse.json
```

Test with a specific checkpoint:
```bash
python3 main.py \
  -e overall_result \
  -a MI3_MedFuse \
  -m test \
  -c ./config/overall_result/MI3/medfuse.json \
  -r /path/to/ckt.pth \
  -b 1
```

Tip: Some configs accept -d GPU_ID (e.g., -d 0).

## 6) Outputs

- Training logs, plots, and checkpoints are saved under ./exp_output/.
- Best checkpoints for quick testing can be stored under:
  - ./exp_output/model_ckt_P12/medfuse/
  - ./exp_output/model_ckt_MI3/medfuse/
  - ./exp_output/model_ckt_HCC/medfuse/

## 7) Troubleshooting

- KeyError: 'HR'
  - Your MI3 CSVs are not in wide format. Ensure they contain vitals columns (HR, O2Sat, etc.) or switch the dataset loader/config to match your schema.
- ValueError: dataset error.
  - The config’s dataset.module/type or dataset.data_path may be incorrect. Verify the path exists and matches your file format.
- CUDA out of memory
  - Reduce batch size in config’s dataloader settings or select a smaller GPU via -d.
- FileNotFoundError for data
  - Align dataset.data_path with your actual folder.

Data access notes:
- MI3 requires a PhysioNet license.
- HCC usage requires IRB approval.


## 8) Other benchmarks

For reproducing results of other benchmark models, refer to:
- https://github.com/sindhura97/STraTS
