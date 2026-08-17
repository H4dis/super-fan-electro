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
