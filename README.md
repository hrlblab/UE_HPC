# UE_HPC

This repository contains the code for Unlearnable Clusters Modified for DDP usage. If you want to run the original method on non-DDP environment, please see [Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples](https://github.com/jiamingzhang94/Unlearnable-Clusters) 

## Setup and Usage

1. **Preprocessing:** Rename datasets and perform initial clustering.
   ```bash
   python preprocess.py --config config/preprocess.yaml -f rename
   ```

2. **Generate Perturbations:** Create cluster-wise perturbations. Note that please modify the configuration as the template provided in `utils/util.py` if you want to use other models as surrogate model.
   ```bash
   python main.py --config config/stage_1.yaml -e {experiment} --stage 1
   ```
3. **Train Models:** Train target models with the generated unlearnable examples. Note that please modify the configuration as the template provided in `utils/util.py` as well if you want to use other models as target model.
   ```bash
   python main.py --config config/stage_2.yaml -e {experiment} --stage 2
   ```
