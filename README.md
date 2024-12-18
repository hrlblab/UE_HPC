# UE_HPC

This repository contains the code for Unlearnable Clusters Modified for DDP usage. If you want to run the original method on non-DDP environment, please see [Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples](https://github.com/jiamingzhang94/Unlearnable-Clusters) 

## Setup and Usage

1. **Preprocessing:** Rename datasets and perform initial clustering.
   ```bash
   python preprocess.py --config config/preprocess.yaml -f rename
   ```

2. **Generate Perturbations:** Create cluster-wise perturbations. Note that please set your environment in (`setup()`) if you want to manually change the ```Bash Master Address```.
   ```bash
   python main.py --config config/stage_1.yaml -e {experiment} --stage 1
   ```
3. **Train Models:** Train target models with the generated unlearnable examples.
   ```bash
   python main.py --config config/stage_2.yaml -e {experiment} --stage 2
   ```
