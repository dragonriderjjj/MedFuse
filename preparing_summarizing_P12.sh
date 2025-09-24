cp preprocess/physionet_2012_toCSV.py medical_ts_datasets/
cd medical_ts_datasets
python3 physionet_2012_toCSV.py
cd ..
python3 preprocess/summarize_P12.py
