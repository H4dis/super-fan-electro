# 🏭 Bearing Fault Diagnosis with Ensemble Learning

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red.svg)](https://xgboost.readthedocs.io/)
[![F1-Score](https://img.shields.io/badge/F1--Score-89.46%25-brightgreen.svg)]()

> **An end-to-end machine learning pipeline for detecting bearing failures using vibration data**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Challenges & Solutions](#challenges--solutions)
- [Contributing](#contributing)
- [License](#license)

---

## 📌 Overview

This project implements a **hybrid ensemble model** for classifying bearing health status into three categories:
- **Class 0**: Healthy
- **Class 1**: Severe Fault
- **Class 2**: Mild Fault

The system combines:
- **Tree-based models** (LightGBM, XGBoost, ExtraTrees, RandomForest)
- **Contextual KNN** (weighted by inverse distance)
- **Adaptive Alpha Blending** (dynamic weighting based on local similarity)

---

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| **F1-Score** | **89.46%** 🏆 |
| **Precision** | 89.11% |
| **Recall** | 89.96% |
| **Accuracy** | 89.72% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0 (Healthy)** | 93.33% | 87.50% | 90.32% | 160 |
| **1 (Severe Fault)** | 89.47% | 91.28% | 90.37% | 149 |
| **2 (Mild Fault)** | 84.54% | 91.11% | 87.70% | 90 |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│ Raw Data (8 features) │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Feature Engineering │
│ • Ratios (4) • Fault Indicators (3) │
│ • Log Transforms (6) • Interactions (3) │
│ • Group Z-Score (6) │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Feature Selection (SelectKBest → 20) │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Scaling (RobustScaler) │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Ensemble Training │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ LightGBM │ │ XGBoost │ │ExtraTrees│ │RandomForest│ │
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│ └─────────────┴─────────────┴─────────────┘ │
│ ▼ │
│ ┌─────────────────────┐ │
│ │ Weighted Voting (Ensemble) │ │
│ └─────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Hybrid Routing (KNN + Ensemble) │
│ • Small groups (≤10) → KNN │
│ • Large groups (>10) → Ensemble │
└─────────────────────────┬───────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ Prediction │
└─────────────────────────────────────────────────────────────┘

text

---
## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

---
## 📊 Results

### 📈 Confusion Matrix

| Actual \ Predicted | Class 0 | Class 1 | Class 2 |
|--------------------|---------|---------|---------|
| **Class 0** | 140 | 10 | 20 |
| **Class 1** | 13 | 136 | 16 |
| **Class 2** | 8 | 15 | 82 |

---

### 🏆 Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **COMP_NAME** | 1558.0 |
| 2 | **MP_LOC** | 859.0 |
| 3 | **Acc, Rms (RMS)** | 821.0 |
| 4 | **Crest (RMS)** | 695.0 |
| 5 | **Crest_Factor** | 651.0 |
| 6 | **Vel, Rms (RMS)** | 633.0 |
| 7 | **Early_Fault_Index** | 605.0 |
| 8 | **Kurt (RMS)** | 599.0 |
| 9 | **Acc_to_Vel_Ratio** | 598.0 |
| 10 | **Kurtosis_x_Acc** | 592.0 |

---

### 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **F1-Score** | **89.46%** 🏆 |
| **Precision** | 89.11% |
| **Recall** | 89.96% |
| **Accuracy** | 89.72% |

---

### 📋 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Class 0 (Healthy)** | 93.33% | 87.50% | **90.32%** | 160 |
| **Class 1 (Severe Fault)** | 89.47% | 91.28% | **90.37%** | 149 |
| **Class 2 (Mild Fault)** | 84.54% | 91.11% | **87.70%** | 90 |

