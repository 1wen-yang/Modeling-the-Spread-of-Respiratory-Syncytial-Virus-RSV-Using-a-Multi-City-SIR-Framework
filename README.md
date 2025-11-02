# Modeling SIR with Travel Coupling (RSV Example)

This project simulates a multi-city SIR model that incorporates commuting interactions 
between Cambridge, South Cambridgeshire, and Huntingdonshire, using population and 
commuting data from the UK Office for National Statistics (ONS).

## Main Features
- Balanced commuting matrix ensuring population conservation
- SIR model with travel coupling
- Lockdown scenario with reduced infection rate (β×0.75 after Day 45)
- Truncation error analysis (O(h²) behavior)
- Richardson extrapolation

## File Structure

Modeling_SIR_travel/
├── main.py                         # Main simulation script
├── ODWP01EW_LTLA.csv               # Commuting data (ONS)
├── mye24tablesuk.xlsx              # Population data (ONS)
├── commuting_matrix_counts_core3.csv
├── commuting_matrix_ratio_core3.csv
└── figs/                           # Folder for generated plots