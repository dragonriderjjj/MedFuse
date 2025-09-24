cp preprocess/mimic_toCSV.py medical_ts_datasets/
cd medical_ts_datasets
python3 mimic_toCSV.py
cd ..
python3 preprocess/summarize_MI3.py
