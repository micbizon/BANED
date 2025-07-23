#!/bin/bash

set -eux

MYSELF_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
source "${MYSELF_DIR}/constants.env"

python prep_data.py -i "${ALL_DATA_FILE}" -o "${ALL_DATA_CLEAN_FILE}"

# Uncomment below two lines if you want to separate fake and real files
python prep_data.py -i "${FAKE_DATA_FILE}" -o "${FAKE_DATA_CLEAN_FILE}"
python prep_data.py -i "${REAL_DATA_FILE}" -o "${REAL_DATA_CLEAN_FILE}"
