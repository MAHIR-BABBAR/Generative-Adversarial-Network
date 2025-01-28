# Generative Adversarial Network (GAN) for Attack Simulation  

This project implements a Generative Adversarial Network (GAN) for simulating attack data using the DNN-EdgeIIoT dataset and evaluates the impact of synthetic data on classification performance.

---

## Features  

### GAN Training:  
- **Generator:** Produces synthetic attack data.  
- **Discriminator:** Evaluates data authenticity.  
- **Training:** Adversarial training for each attack type, saving trained models and scalers.  

### Synthetic Data Generation:  
- Loads pre-trained GAN models for generating synthetic attack data.  
- Scales synthetic data back to its original range using saved scalers.  
- Combines data from multiple attack types into a unified synthetic dataset.  
- Saves the synthetic dataset as `synthetic_dataset.csv`.  

### Evaluation of Synthetic Data:  
- Combines synthetic and original datasets to assess model performance improvements.  
- Uses **Random Forest Classifier** for evaluation through 5-fold cross-validation.  
- Compares model accuracy:  
  - **Original Data:** Accuracy using only real data.  
  - **Combined Data:** Accuracy using real + synthetic data.  

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/MAHIR-BABBAR/Generative-Adversarial-Network.git  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---

## Usage  

### Training GAN:  
Train GAN models on each attack type in the DNN-EdgeIIoT dataset and save the models and scalers.  

### Generating Synthetic Data:  
1. Load pre-trained models and scalers.  
2. Generate synthetic data for each attack type.  
3. Combine and save the synthetic dataset.  

### Evaluation:  
1. Load the original and synthetic datasets.  
2. Evaluate classification accuracy using k-fold cross-validation.  
3. Compare results to analyze the impact of synthetic data.  

---

## Example Results  

| **Dataset**              | **Accuracy** |  
|---------------------------|--------------|  
| Original Data             | 88.45%        |  
| Original + Synthetic Data | 97.19%        |  

---

## Contact  

**Author:** Mahir Babbar  
**GitHub:** [MAHIR-BABBAR](https://github.com/MAHIR-BABBAR)  

---

