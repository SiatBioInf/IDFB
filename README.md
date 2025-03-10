# IDFB: Integration of Data From Bulk RNA Sequencing Platforms

## Introduction
IDFB (Integration of Data From Bulk RNA Sequencing Platforms) is a deep learning framework designed to eliminate batch effects and platform-specific variations in RNA sequencing data while preserving essential biological information. By constructing pseudo-samples from multi-platform data of cell lines and immune cells, and leveraging these synthetic data for model training, we developed a GAN-based model that can effectively remove platform-specific biases while maintaining biological relevance.


## Key Features
- **Platform-Agnostic Integration**: Eliminates technical variations across different RNA sequencing platforms
- **Biological Information Preservation**: Maintains critical biological signals during the integration process
- **Comprehensive Evaluation System**: Assesses both biological conservation and platform mixing through multiple metrics
- **Versatile Applications**: Validated through various downstream tasks:
  - Cancer Type Classification
  - Lung Cancer Subtype Classification  
  - Survival Analysis
  - MSI Status Prediction


## Methodology
1. **Data Collection**: Gathered RNA-seq data from multiple platforms including tumor cell lines and immune cells to construct multiple sets of pseudo-samples
2. **Model Training**: Utilized GAN architecture to learn platform-invariant representations, the structure of which is shown in the figure below:
   ![Model Architecture](https://github.com/IDFB/blob/main/model_architecture.jpg) 
3. **Data Generation**: Generated integrated data with reduced platform effects
4. **Evaluation**: 
   - Biological Conservation Metrics:
     - Mean Average Precision (MAP)
     - Average Silhouette Width (ASW) 
     - Neighborhood Consistency (NC)
   - Platform Mixing Assessment:
     - Seurat Alignment Score (SAS)
     - GPL Average Silhouette Width (GPL-ASW)
     - Graph Connectivity (GC)


## Performance
The framework demonstrates robust performance in:
- Removing platform-specific variations
- Maintaining biological information
- Improving downstream analysis accuracy
- Facilitating cross-platform data integration


## Project Structure
IDFB/
├── Dataset/                        # Data directory
│   ├── header.txt                  # Feature list
│   ├── MSI/                        # MSI dataset
│   ├── Lung_cancer/                # Lung cancer dataset
│   ├── Survival/                   # Survival analysis dataset
│   └── Cancertype/                 # Cancer type dataset
├── src/                            # Source code
│   ├── generate_pseudo_data.py     # Generate pseudo-samples
│   ├── process_data.py             # Generate training sets and test sets
│   ├── models.py                   # Model architecture
│   ├── train.py                    # Training pipeline
│   ├── integrate.py                # Data integration
│   ├── cancertype_data_process.py  # Process the cancer type dataset
│   ├── lungcancer_data_process.py  # Process the lung cancer dataset
│   ├── msi_data_process.py         # Process the MSI dataset
│   ├── survival_data_process.py    # Process the survival analysis dataset
│   └── evaluate.py                 # Evaluation metrics
└── saved_models/                   # Saved model checkpoints


## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+

### Setup
```bash
# Clone the repository
git clone https://github.com/SiatBioInf/IDFB.git
cd IDFB

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data
1. Create a new task directory in the Dataset folder
2. Process your data into a `processed_data.csv` file with the following format:
   - Columns aligned with `header.txt`
   - Second-to-last column: platform labels encoded as:
     ```python
     platform_encoder = {
         'GPL570': 0,
         'GPL20301': 1, 
         'GPL24676': 2,
         'unknown': 3
     }
     ```
   - Last column: downstream task labels

### 2. Generate Integrated Data
Run the integration script with your task name:
```bash
python src/integrate.py --task [your_task_folder_name]
```
The integrated data will be saved as generated_data.csv in your task folder.

### 3. Evaluate Results
Assess the quality of integration:
```bash
python src/evaluate.py --task [your_task_folder_name]
```
This will output:
- Biological conservation score
- Platform mixing score

Note: Replace [your_task_folder_name] with the name of your task directory created in step 1.


## Example
```bash
# Generate integrated data
python IDFB/integrate.py --task MSI
# Evaluate results
python IDFB/evaluate.py --task MSI
```

## More
More information could be found in our paper: Zihao Li, Rui Zhang, Zhen Wang, Zhilan Xu, Huahui Li, Xiaomin Ni, Xuefei Lin, Yan Zhang, Yang Zhang, Hongyan Wu, Hao Yu. A GAN-VAE model for the integration of bulk transcriptomic data across different sequencing platforms(http://)(Submitting).
