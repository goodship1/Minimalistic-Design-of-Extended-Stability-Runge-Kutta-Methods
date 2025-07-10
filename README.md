# Minimalistic Design of Extended Stability Runge-Kutta (ESRK) Methods.

This repository contains research code for a potential new **Extended Stability Runge-Kutta (ESRK)** scheme. The design leverages a novel structure that requires only **2S-8 unique coefficients**, significantly reducing memory and computational requirements while maintaining high accuracy and stability. This structure has been tested when S>=15 where S is the number of stages in the butcher tableau.

## Overview

Traditional ESRK methods often involve high storage requirements and computational redundancy. This repository presents a **minimalistic ESRK design** that exploits repeated coefficients and unique stage representations to optimize both storage and computational efficiency.

### Key Features

1. **Low Storage Requirements**:
   - The scheme uses a novel structure requiring only **2S-7 unique coefficients**, reducing the memory footprint compared to traditional ESRK schemes. S is the number of stages this reflects the number of unique a-values which are needed 
.

3. **Generalized Structure**:
   - The design can be applied to **3rd-order 15 and 21 stage**, **4th-order 16-stage**, and potentially other high-order ESRK methods.

4. **Improved Computational Efficiency**:
   - By combining **compressed representation** and **precomputation**, the scheme achieves faster runtimes and reduced memory usage.

5. **Potential Applications**:
   - Suitable for solving stiff and mildly stiff ODEs, particularly in resource-constrained environments with critically low-storage methods.

6. Fifth order root trees order conditions checked 2S-6 unique coefficients are able to meet the rooted tree conditions when S>=12, but when extended stability requirements, S has to be greater than 18 

---

## Repository Contents

### 1. **Codebase**
   - Implementation of the **2S-8 ESRK scheme** for testing and validation.
   - Includes examples for 3rd-order 15 and 21-stage and 4th-order 16-stage Butcher tableaus.
   - Ipopt code which is used to generated a search of the butcher tableaus along with the dockerfile used to run them locally.

### 2. **Order Condition Verification**
   - Automated checks to ensure the scheme satisfies required order conditions.

### 3. **Performance Analysis**
   - Scripts for benchmarking against traditional ESRK methods.
   - Metrics include runtime, memory usage, and convergence behaviour.

### 4. **Stability Analysis**
   - Tools to analyze the stability region of the proposed scheme along with the stability polynomial structure. All the tableaus used in the paper are stability polynomial and calculations and plots are provided in there own seperate notebook. This includes the 3rd order 15 , 21 stage and 4th order 16 stage.
### 5. **Collect of butcher tableaus**
   -  Also in the repo is a collection of 3rd order 4th order tableaus for researchers to use.
   -  Also the docker file to run the IPOPT code to generate the schemes available for other researchers to generate their own schemes.
   -  For the longer running 21 stage  esrk i have supplied some tableaus but feel free to generate your own all IPOPT code is provided. 
     

---

## Usage
Download the dockerfule to run the ipopt code with the constraints added in to generate the tableaus for analysis the third order 21 stage takes a while to find a convergence solution and reference butcher tableaus will be provided for the tableaus in the paper which is attached with this code. 

### Requirements
- Python 3.8+
- Dependencies:
  - `numpy`
  - `scipy`
  - `matplotlib`

Install the required dependencies using:
```bash
pip install -r requirements.txt
