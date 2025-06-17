
# 🛡️ FIT5124 Assignment 3 – Model Extraction & Membership Inference Attacks

This repository contains my submission for **FIT5124 Assignment 3**, which explores two major machine learning security threats:

- **Model Extraction Attacks (MEA)**
- **Membership Inference Attacks (MIA)**

The project includes implementation, evaluation, and defence strategies for both types of attacks using a PyTorch-based MNIST classifier.

---

## 📂 Project Structure

### 🔧 Core Model Files

- `a3_mnist.py`  
  Original training script for the MNIST model.

- `a3_mnist_fixed.py`  
  Improved version of the training script with dropout and early stopping.

- `target_model.pth`  
  Trained target MNIST model used in all attack experiments.

---

### 🎯 Model Extraction Attack (MEA)

- `model_extraction_attack.py`  
  Baseline model extraction script. Queries the target model and trains a surrogate.

- `model_extraction_attack_defended.py`  
  Runs model extraction with defences (e.g., noisy labels, confidence masking).

- `defended_model_server.py`  
  Contains defence functions: label flipping, confidence noise, adaptive outputs.

- `extracted_model.pth`  
  Surrogate model trained on unprotected outputs.

- `extracted_model_defend1.pth`  
  Surrogate model trained on top-1 label only defence.

- `extracted_model_strong_defence.pth`  
  Surrogate model trained using adaptive defence (label flip + noise).

---

### 🔐 Membership Inference Attack (MIA)

- `membership_inference_attack.py`  
  Implements MIA on the target model using full confidence vectors and a binary classifier.

---

### ✅ Evaluation Scripts

- `test_attack.py`  
  Compares surrogate model accuracy and agreement with the target model.

- `test_setup.py`  
  Tests model loading and dummy predictions for verification.

---

## ▶️ How to Run

````

````
### 1. Train Target Model

```bash
python a3_mnist_fixed.py
```

### 2. Run Model Extraction (Baseline)

```bash
python model_extraction_attack.py
```

### 3. Run Model Extraction with Defence

```bash
python model_extraction_attack_defended.py
```

### 4. Evaluate Surrogate Model

```bash
python test_attack.py
```

### 5. Run Membership Inference Attack

```bash
python membership_inference_attack.py
```

---

## 📌 Notes

* All models are trained using the **MNIST** dataset with standard transforms and architecture.
* Defences are applied externally (as per assignment instructions) and do not modify the training process.
* Metrics used include model accuracy, prediction agreement, and attack model success rate.

---

## 📚 References

* Shokri, R., et al. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S\&P.
* Tramèr, F., et al. (2016). Stealing Machine Learning Models via Prediction APIs. USENIX Security.
* Abadi, M., et al. (2016). Deep Learning with Differential Privacy. ACM CCS.
* PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

## 👩‍💻 Author

**Vanshika Kashettiwar** 
Master of Cybersecurity – Monash University
FIT5124

