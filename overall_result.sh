#!/usr/bin/env bash
mode=$1
dataset=$2

if [ "$mode" = "train" ]; then
    model="MedFuse"
    if [ "$dataset" = "HCC" ]; then
        echo "Warning: This dataset can be access unless you have IRB approval."
        echo ""
        echo "Train ${model} on ${dataset}."
        echo ""
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "train" -c "./config/overall_result/HCC/medfuse.json" -d 1
    elif [ "$dataset" = "MI3" ]; then
        echo "Warning: You have to get the approval before accessing the dataset. More detail in: https://physionet.org/content/mimiciii/"
        echo ""
        echo "Train ${model} on ${dataset}."
        echo ""
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "train" -c "./config/overall_result/MI3/medfuse.json"
    elif [ "$dataset" = "P12" ]; then
        echo ""
        echo "Train ${model} on ${dataset}."
        echo ""
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "train" -c "./config/overall_result/P12/medfuse.json"
    else
        echo "dataset should be \"P12\" , \"MI3\", or \"HCC\"."
    fi
elif [ "$mode" = "test" ]; then
    model="MedFuse"
    echo "This mode will use the provided checkpoint (trained by the author) to test on the dataset."
    if [ "$dataset" = "HCC" ]; then
        echo "Test ${model} on ${dataset}."
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "test" -c "./exp_output/model_ckt_HCC/medfuse/config.json" -r "./exp_output/model_ckt_HCC/medfuse/ckt.pth" -b 1
    elif [ "$dataset" = "MI3" ]; then
        echo "Warning: You have to get the approval before accessing the dataset. More detail in: https://physionet.org/content/mimiciii/"
        echo "Test ${model} on ${dataset}."
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "test" -c "./exp_output/model_ckt_MI3/medfuse/config.json" -r "./exp_output/model_ckt_MI3/medfuse/ckt.pth" -b 1
    elif [ "$dataset" = "P12" ]; then
        echo "Test ${model} on ${dataset}."
        python3 main.py -e "overall_result" -a "${dataset}_${model}" -m "test" -c "./exp_output/model_ckt_P12/medfuse/config.json" -r "./exp_output/model_ckt_P12/medfuse/ckt.pth" -b 1
    else
        echo "dataset should be \"P12\" , \"MI3\", or \"HCC\"."
    fi
else
    echo "mode should be \"train\" or \"test\""
fi