
# **GWSolver**
Groundwater Solver for Darcy Flow in Porous Media.

---

## **Table of Contents**
1. [Repository Usage](#repository-usage)
2. [Environment Setup](#environment-setup)
3. [Test Data](#test-data)
4. [Testing Instructions](#testing-instructions)

---

## **Repository Usage**
- This project uses the `dev` branch for active development.  
  Ensure you switch to the `dev` branch before making any changes:
  ```bash
  git checkout dev
  ```

---

## **Environment Setup**
- To set up the required environment, refer to the [`environment.yml`](./environment.yml) file.
- Use the following command to create the environment:
  ```bash
  conda env create -f environment.yml
  ```

---

## **Test Data**
- Download the required test data from the following link:  
  [Test Data Folder](https://drive.google.com/drive/folders/1_8SIsaUw16l-K4Jw-j6atwFFgIJmCYDR?usp=sharing)
  
  **Files:**
  - `benchmark_1024.mat`: For testing in `gwsolver_2D_steady_state.py`.
  - `benchmark_1024_transient.mat`: For testing in `gwsolver_2D_transient.py`.

---

## **Testing Instructions**
1. Place the downloaded test data in the appropriate directory of your project.
2. Use the corresponding Python scripts for testing:
   - **Steady State Solver:**
     ```bash
     python gwsolver_2D_steady_state.py
     ```
   - **Transient Solver:**
     ```bash
     python gwsolver_2D_transient.py
     ```

---

Feel free to suggest improvements or raise issues for better collaboration. 🚀
