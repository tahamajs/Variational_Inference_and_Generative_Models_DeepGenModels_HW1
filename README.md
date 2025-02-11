# ELBO_KL_Bayesian_VAE_PGM_HW1

# Deep Generative Models - Homework 1

## 📌 Table of Contents

- [Introduction](#introduction)
- [Course Information](#course-information)
- [Assignment Details](#assignment-details)
- [Sections Overview](#sections-overview)
  - [PGM (Probabilistic Graphical Models)](#pgm-probabilistic-graphical-models)
  - [VAE (Variational Autoencoders)](#vae-variational-autoencoders)
- [Implementation Details](#implementation-details)
- [Mathematical Derivations](#mathematical-derivations)
- [Training and Experimentation](#training-and-experimentation)
- [Submission Guidelines](#submission-guidelines)
- [Academic Integrity Policy](#academic-integrity-policy)
- [License](#license)

---

## 📝 Introduction

This repository contains **Homework 1** for the **Deep Generative Models** course at the **University of Tehran**. The homework consists of **theoretical and practical problems** in **probabilistic graphical models (PGMs)** and **variational autoencoders (VAEs)**. The goal is to:

- Develop a strong mathematical understanding of **Bayesian Networks**, **Markov Networks**, and **inference techniques**.
- Implement and train **Variational Autoencoders (VAEs)** using deep learning frameworks.
- Explore **latent space properties**, **disentanglement**, and **factorized representations**.

By completing this assignment, students will gain **both theoretical and practical knowledge** necessary for working with **deep generative models**.

---

## 🎓 Course Information

- **University**: University of Tehran
- **Department**: Electrical and Computer Engineering
- **Course**: Deep Generative Models
- **Instructor**: Dr. Mostafa Tavasoli
- **Term**: Fall 1403

---

## 🏆 Assignment Details

This assignment is **divided into two main sections**:

### 🔹 **1. Probabilistic Graphical Models (PGM)**:

- Understanding and constructing **Bayesian Networks** and **Markov Networks**.
- Deriving **conditional independence properties** and factorized probability distributions.
- Implementing **variational inference** for a Bayesian Network.

### 🔹 **2. Variational Autoencoders (VAE)**:

- **Theoretical Derivations**: Understanding ELBO, KL Divergence, and reparameterization.
- **Implementation**: Coding a **VAE model** using **deep learning frameworks**.
- **Experiments**: Training a VAE, visualizing the latent space, and analyzing disentanglement.

---

## 📂 Sections Overview

### 🔥 **PGM (Probabilistic Graphical Models)**

Probabilistic Graphical Models (PGMs) are **powerful tools** for modeling complex probability distributions using graphs.

#### ✅ **Tasks:**

1. **Bayesian Network for Server Monitoring**:

   - Define a Bayesian Network for a **server monitoring system** where temperature, air conditioning failure, and open doors affect server behavior.
   - Construct a **directed acyclic graph (DAG)** for this problem.
   - Compute the **joint probability distribution**.
2. **Independence Properties**:

   - Verify **conditional independence statements** using d-separation properties.
3. **Markov Network Analysis**:

   - Analyze a **Markov network**, verify independence properties, and factorize distributions.
4. **Variational Inference**:

   - Implement **variational inference** to estimate posterior distributions.

---

### 🔥 **VAE (Variational Autoencoders)**

Variational Autoencoders (VAEs) are generative models that **learn probabilistic latent space representations**.

#### ✅ **Tasks:**

1. **ELBO Derivation**:

   - Prove the **Evidence Lower Bound (ELBO)** equation.
   - Show how **KL divergence** regularizes latent space.
2. **Reparameterization Trick**:

   - Implement the **Reparameterization Trick** to enable gradient-based optimization.
3. **Building the VAE Model**:

   - Implement an **Encoder-Decoder structure**.
   - Use **convolutional layers** for feature extraction.
4. **Training and Experimentation**:

   - Train the VAE on an **image dataset**.
   - Evaluate the model using **reconstruction loss and KL divergence**.
   - Visualize the latent space and generate **new images**.
5. **β-VAE and Disentanglement**:

   - Implement **β-VAE** to study **factorized latent representations**.
   - Experiment with modifying latent variables to control image attributes.

---

## ⚙️ Implementation Details

### **🔹 Dataset**

- The dataset contains **face images**.
- **Training/Test split**: **80/20**.
- **Preprocessing**:
  - Convert images to grayscale (if required).
  - Resize images to **128x128**.
  - Normalize pixel values to `[0,1]`.

### **🔹 Model Architecture**

The **VAE architecture** consists of:

| Component         | Layers Used                                                  |
| ----------------- | ------------------------------------------------------------ |
| **Encoder** | Conv2D (ReLU) → Conv2D → Flatten → Dense (Latent Space)   |
| **Decoder** | Dense → Reshape → Conv2D Transpose (ReLU) → Conv2D Output |

### **🔹 Training Parameters**

| Parameter     | Value      |
| ------------- | ---------- |
| Image Size    | (128, 128) |
| Batch Size    | 128        |
| Optimizer     | Adam       |
| Learning Rate | 0.0005     |
| Epochs        | 1000       |

### **🔹 Loss Function**

- **ELBO Loss**: Combination of **Reconstruction Loss** and **KL Divergence**.
- **β-VAE Loss**: Adds a weight term `β` to KL Divergence for disentanglement.

---

## 📊 Mathematical Derivations

### **1️⃣ ELBO Derivation**

The **ELBO (Evidence Lower Bound)** is derived from **Bayes' Theorem**:

\$$logp(x)=E

$$


$$

$$


$$

**2️⃣ Reparameterization Trick**

Since sampling from \( q(z|x) \) is **non-differentiable**, we use:

\[
z = \mu + \sigma \cdot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0,1)
\]

This enables **gradient-based training**.

---

## 🚀 Training and Experimentation

1. **Train the VAE model** for **1000 epochs**.
2. **Monitor ELBO loss** and plot the **KL divergence** during training.
3. **Visualize latent space** using **interpolations**.
4. **Modify latent vectors** to control **face attributes**.

---

### **📜 Required Files**

1. **Report**: Must include **theoretical answers, derivations, and analysis**.
2. **Code**: Fully executable **Python scripts**.
3. **Submission Format**:
