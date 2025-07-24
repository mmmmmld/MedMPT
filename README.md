# MedMCPT

This repository is a pre-release version of the official implementation of our paper titled "A vision-language pretrained transformer for versatile clinical respiratory disease applications". 
## Usage

### Dependencies
- Linux
- CUDA
- Python = 3.6
- PyTorch = 1.10.2

### Installation
```
$ pip install -r requirements.txt
```

### Data
1. NLST: Access to the data from NLST study must to be approved. Please follow the instructions on the [NLST](https://cdas.cancer.gov/learn/nlst/instructions/).
2. MosMedData: [MosMedData](https://github.com/michaelwfry/MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings/)
3. The data for pretraining and downstream tasks including report generation and prescription recommendation are not publicly available due to privacy requirements. Instead, a small demo dataset will be released to facilitate reproducibility and further research.

### Model weights
The pretrained weights of MedMPT can be accessed upon request via [weights](https://huggingface.co/liangdi123/MedMPT) for non-commercial research and academic use.
You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author listed in the paper.

### Pretraining
Only multi-gpu, DistributedDataParallel training is supported.
To do pretraining in two 4-gpu machines, run the command in each machine:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnodes=2 --node_rank=${node_rank} --master_addr=${master_addr} --master_port=${master_port} --pretrain.py --exp_name pretrain
```
by replacing the ${node_rank} with the rank of current machine (i.e., 0 or 1), ${master_addr} with the ip address of the machine rank 0, and ${master_port} with an available port id in the machine rank 0.

### Fine-tuning with MedMPT pretrained weights
Replacing the ${path_to_pretrained_model_parameters} by the path of the pretrained parameters to load pretrained model during finetuning.
1. Lung cancer screening
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main_nlst.py --exp_name lung_cancer_screening --vision_load_dir ${path_to_pretrain_model_parameters}
```

2. COVID-19 identifying
```
CUDA_VISIBLE_DEVICES=0 python main_mosmeddata.py --task diagnosis --exp_name covid19_identifying --vision_load_dir ${path_to_pretrain_model_parameters}
```

3. Severity grading
```
CUDA_VISIBLE_DEVICES=1 python main_mosmeddata.py --task stage --exp_name severity_grading --vision_load_dir ${path_to_pretrain_model_parameters}
```

4. Report generation
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main_caption.py --exp_name report_generation --vision_load_dir_ind ${path_to_pretrained_model_parameters} --caption_load_dir_ind ${path_to_pretrained_model_parameters}
```

5. Prescription recommendation
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_medicine_vit.py --input ct,report,biomarker --vision_load_dir_ind ${path_to_pretrained_model_parameters} --text_load_dir_ind ${path_to_pretrained_model_parameters}
```
