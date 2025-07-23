### Requirements
Make sure you have datasets:
* `fnn_fake.csv`
* `fnn_real.csv`

Install all python dependencies:
```commandline
pip install -r requirements.txt
```

### Scripts
Inside `scripts` directory, there are scripts for running the whole pipeline:
* `constants.env` - set up all the variables to correspond your environment
* `preprocess_data.sh` - cleaning up datasets and creating a knowledge base
* `pipeline.sh` - running an apriori algorithm and training an CNN model

---
### Running python scripts directly

#### Prepare clean dataset
```commandline
python prep_data.py -i fnn_real.csv -o fnn_real_clean.csv
python prep_data.py -i fnn_fake.csv -o fnn_fake_clean.csv
```

#### Apriori algorithm
```commandline
python apriori_algo.py -i fnn_real_clean.csv
```

#### CNN algorithm
```commandline
python cnn.py -r gossipcop_real_clean.csv -f gossipcop_fake_clean.csv
```

---
### DGX
Run the script on DGX, by submitting a job from a script:
`sbatch script.sh`

Example script:
```shell
#!/bin/bash
#SBATCH --job-name=uncertainty-quantification
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=output.out
#SBATCH --error=errors.err

set -e
eval "$(conda shell.bash hook)"
conda activate <your env name>

cd <your project path>

./scripts/preprocess_data.sh
./scripts/pipeline.sh
```

### Calculate support

Use `./calculate.py` to calculate the support. An example command could be:

```bash
./calculate.py Quantification_issues/fnn_all_clean.csv --probabilities Quantification_issues/fnn_all_clean_cnn_prob.npy --fake_support Quantification_issues/fnn_fake_clean_apriori_sup_0.1.csv --real_support Quantification_issues/fnn_real_clean_apriori_sup_0.1.csv --limit 20
```

For more options, run `./calculate.py -h`
