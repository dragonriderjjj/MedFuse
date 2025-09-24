impute_value_1=$1
impute_value_2=$2
impute_value_3=$3

echo "Impute missing AFP with ${impute_value_1} and test by the Random Forest"
python3 impute_robust.py -e "robust_impute" -a "rf_${impute_value_1}" -m "test" -c "config/imputation_robust/random_forest.json" -r "exp_output/model_ckt_P12/rf/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the Random Forest"
python3 impute_robust.py -e "robust_impute" -a "rf_${impute_value_2}" -m "test" -c "config/imputation_robust/random_forest.json" -r "exp_output/model_ckt_P12/rf/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the Random Forest"
python3 impute_robust.py -e "robust_impute" -a "rf_${impute_value_3}" -m "test" -c "config/imputation_robust/random_forest.json" -r "exp_output/model_ckt_P12/rf/ckt.pth" -i ${impute_value_3}

echo ""

echo "Impute missing AFP with ${impute_value_1} and test by the XGBoost"
python3 impute_robust.py -e "robust_impute" -a "xgb_${impute_value_1}" -m "test" -c "config/imputation_robust/xgboost.json" -r "exp_output/model_ckt_P12/xgb/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the XGBoost"
python3 impute_robust.py -e "robust_impute" -a "xgb_${impute_value_2}" -m "test" -c "config/imputation_robust/xgboost.json" -r "exp_output/model_ckt_P12/xgb/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the XGBoost"
python3 impute_robust.py -e "robust_impute" -a "xgb_${impute_value_3}" -m "test" -c "config/imputation_robust/xgboost.json" -r "exp_output/model_ckt_P12/xgb/ckt.pth" -i ${impute_value_3}

echo ""

echo "Impute missing AFP with ${impute_value_1} and test by the GRU"
python3 impute_robust.py -e "robust_impute" -a "gru_${impute_value_1}" -m "test" -c "config/imputation_robust/gru.json" -r "exp_output/model_ckt_P12/gru/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the GRU"
python3 impute_robust.py -e "robust_impute" -a "gru_${impute_value_2}" -m "test" -c "config/imputation_robust/gru.json" -r "exp_output/model_ckt_P12/gru/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the GRU"
python3 impute_robust.py -e "robust_impute" -a "gru_${impute_value_3}" -m "test" -c "config/imputation_robust/gru.json" -r "exp_output/model_ckt_P12/gru/ckt.pth" -i ${impute_value_3}

echo ""

echo "Impute missing AFP with ${impute_value_1} and test by the GRUD"
python3 impute_robust.py -e "robust_impute" -a "grud_${impute_value_1}" -m "test" -c "config/imputation_robust/grud.json" -r "exp_output/model_ckt_P12/grud/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the GRUD"
python3 impute_robust.py -e "robust_impute" -a "grud_${impute_value_2}" -m "test" -c "config/imputation_robust/grud.json" -r "exp_output/model_ckt_P12/grud/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the GRUD"
python3 impute_robust.py -e "robust_impute" -a "grud_${impute_value_3}" -m "test" -c "config/imputation_robust/grud.json" -r "exp_output/model_ckt_P12/grud/ckt.pth" -i ${impute_value_3}

echo ""

echo "Impute missing AFP with ${impute_value_1} and test by the original transformer encoder"
python3 impute_robust.py -e "robust_impute" -a "te_${impute_value_1}" -m "test" -c "config/imputation_robust/te.json" -r "exp_output/model_ckt_P12/te/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the original transformer encoder"
python3 impute_robust.py -e "robust_impute" -a "te_${impute_value_2}" -m "test" -c "config/imputation_robust/te.json" -r "exp_output/model_ckt_P12/te/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the original transformer encoder"
python3 impute_robust.py -e "robust_impute" -a "te_${impute_value_3}" -m "test" -c "config/imputation_robust/te.json" -r "exp_output/model_ckt_P12/te/ckt.pth" -i ${impute_value_3}

echo ""

echo "Impute missing AFP with ${impute_value_1} and test by the TranSCANE"
python3 impute_robust.py -e "robust_impute" -a "tesne_${impute_value_1}" -m "test" -c "config/imputation_robust/tesne.json" -r "exp_output/model_ckt_P12/tesne/ckt.pth" -i ${impute_value_1}
echo "Impute missing AFP with ${impute_value_2} and test by the TranSCANE"
python3 impute_robust.py -e "robust_impute" -a "tesne_${impute_value_2}" -m "test" -c "config/imputation_robust/tesne.json" -r "exp_output/model_ckt_P12/tesne/ckt.pth" -i ${impute_value_2}
echo "Impute missing AFP with ${impute_value_3} and test by the TranSCANE"
python3 impute_robust.py -e "robust_impute" -a "tesne_${impute_value_3}" -m "test" -c "config/imputation_robust/tesne.json" -r "exp_output/model_ckt_P12/tesne/ckt.pth" -i ${impute_value_3}
