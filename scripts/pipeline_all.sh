#!/bin/bash

set -eux

MYSELF_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source "${MYSELF_DIR}/constants.env"

python apriori_algo.py -i "${ALL_DATA_CLEAN_FILE}"

supports=(
    "${DATA_DIR}/results/all_clean_apriori_sup_0.1.csv"
    "${DATA_DIR}/results/all_clean_apriori_sup_0.2.csv"
    "${DATA_DIR}/results/all_clean_apriori_sup_0.3.csv"
    "${DATA_DIR}/results/all_clean_apriori_sup_0.4.csv"
)

mc_probabilities=(
    0.2
    0.3
    0.4
    0.5
)

bp_probabilities=(
    0.2
    0.3
    0.4
    0.5
)

# measure time
printf "options,elapsed_time,elapsed_time_inference\n" >"${TIME_RESULTS_FILE}"
python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -e -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
for mc_prob in "${mc_probabilities[@]}"; do
    python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -dp "${mc_prob}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
done
for bc_prob in "${bp_probabilities[@]}"; do
    python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -bp "${bc_prob}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
done
for supp_file in "${supports[@]}"; do
    python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -kb "${supp_file}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
    python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -kb "${supp_file}" -e -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
    for mc_prob in "${mc_probabilities[@]}"; do
        python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -dp "${mc_prob}" -kb "${supp_file}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
    done
    for bc_prob in "${bp_probabilities[@]}"; do
        python fit_transform.py -r "${REAL_DATA_CLEAN_FILE}" -f "${FAKE_DATA_CLEAN_FILE}" -bp "${bc_prob}" -kb "${supp_file}" -t "${TIME_RESULTS_FILE}" -m "${MODEL_TYPE}"
    done
done
