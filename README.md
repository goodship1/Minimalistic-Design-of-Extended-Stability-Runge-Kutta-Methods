# Minimalistic Design of Extended Stability Runge-Kutta (ESRK) Methods.

This repository contains research code for a potential new **Extended Stability Runge-Kutta (ESRK)** scheme. The design leverages a novel structure that requires only **2S-7 unique coefficients**, significantly reducing memory and computational requirements while maintaining high accuracy and stability. This structure has been tested when S>=15 where S is the number of stages in the butcher tableau.

## Overview

Traditional ESRK methods often involve high storage requirements and computational redundancy. This repository presents a **minimalistic ESRK design** that exploits repeated coefficients and unique stage representations to optimize both storage and computational efficiency.

### Key Features

1. **Low Storage Requirements**:
   - The scheme uses a novel structure requiring only **2S-7 unique coefficients**, reducing the memory footprint compared to traditional ESRK schemes. S is the number of stages this reflects the number unique a values which are needed 
.

3. **Generalized Structure**:
   - The design can be applied to **3rd-order 15 and 21 stage**, **4th-order 16-stage**, and potentially other high-order ESRK methods.

4. **Improved Computational Efficiency**:
   - By combining **compressed representation** and **precomputation**, the scheme achieves faster runtimes and reduced memory usage.

5. **Potential Applications**:
   - Suitable for solving stiff and mildly stiff ODEs, particularly in resource-constrained environments with critical low-storage methods.

---

## Repository Contents

### 1. **Codebase**
   - Implementation of the **2s-7 ESRK scheme** for testing and validation.
   - Includes examples for 3rd-order 15-stage and 4th-order 16-stage Butcher tableaus.
   - Ipopt code which is used to generated a search of the butcher tableaus along with the dockerfile used to run them locally.

### 2. **Order Condition Verification**
   - Automated checks to ensure the scheme satisfies required order conditions.

### 3. **Performance Analysis**
   - Scripts for benchmarking against traditional ESRK methods.
   - Metrics include runtime, memory usage, and convergence behavior.

### 4. **Stability Analysis**
   - Tools to analyze the stability region of the proposed scheme along with the stability polynomial structure.
### 5. **Collect of butcher tableaus**
   -  Also in the repo is a collection of 3rd order 4th order tableaus for researchers to use.
   -  Also the docker file to run the IPOPT code to generate the schemes available for other researchers to generate there own schemes  
     

---

## Usage

### Requirements
- Python 3.8+
- Dependencies:
  - `numpy`
  - `scipy`
  - `matplotlib`

Install the required dependencies using:
```bash
pip install -r requirements.txt
