# Modeling the Spread of Respiratory Syncytial Virus (RSV) Using a Multi-City SIR Framework

This project simulates a **multi-city SIR epidemiological model** that incorporates daily commuting
interactions between **Cambridge**, **South Cambridgeshire**, and **Huntingdonshire**.
It uses official population and commuting data from the **UK Office for National Statistics (ONS)**,
and models the transmission dynamics of **Respiratory Syncytial Virus (RSV)**.

---

## Main Features
- Balanced commuting matrix ensuring population conservation  
- SIR model with explicit travel coupling  
- Lockdown scenario with reduced infection rate (β × 0.75 after Day 45)  
- Truncation error analysis demonstrating O(h²) convergence  
- Richardson extrapolation and error comparison with reference solution  

---

## File Structure
```
Modeling-the-Spread-of-Respiratory-Syncytial-Virus-RSV-Using-a-Multi-City-SIR-Framework/
├── main.py                         # Main simulation script
├── ODWP01EW_LTLA.csv               # Commuting data (ONS)
├── mye24tablesuk.xlsx              # Population data (ONS)
├── commuting_matrix_counts_core3.csv
├── commuting_matrix_ratio_core3.csv
└── figs/                           # Folder for generated plots
```

---

## How to Run
1. Open a terminal and navigate to the project directory:
   ```bash
   cd ~/Desktop/Modeling_SIR_travel
   ```

2. Run the simulation:
   ```bash
   python3 main.py
   ```

3. The program will automatically generate several plots:
   - Infection curves (Baseline vs Lockdown)
   - Susceptible and Recovered populations  
   - Truncation error curve  
   - Richardson extrapolation error plot  