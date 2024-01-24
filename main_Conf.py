from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import json

fp = json.load(open('parameters/scenarios15m.json', 'r'))
par = json.load(open('parameters/parameters.json', 'r'))
contingency = json.load(open('parameters/contingency.json', 'r'))

Ωa = json.load(open('parameters/EV.json', 'r'))
Ωt = list(range(1, 97))
Ωc = contingency['timestamp']
Ωs = fp.keys()
Δt = 0.25  # Define the time interval in hours

πc = {contingency['timestamp'][i]: contingency['probability'][i]  for i in range(0,len(Ωc))}  # Define the probability of each contingency
πs = {s: fp[s]['prob'] for s in Ωs}  # Define the probability of each scenario

# Create a Gurobi model
model = gp.Model(par['model name'])

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0)
PGDmax = model.addVar(name="PGDmax", lb=0)
PAEmax = model.addVar(name="PAEmax", lb=0)
EAEmax = model.addVar(name="EAEmax", lb=0)

PS   = {(t, s, c, a): model.addVar(name=f"PS_{t}_{c}_{a}", lb=par['PSmin'], ub=par['PSmax'])    for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSp   = {(t, s, c, a): model.addVar(name=f"PSp_{t}_{c}_{a}", lb=0, ub=par['PSmax'])             for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSn   = {(t, s, c, a): model.addVar(name=f"PSn_{t}_{c}_{a}", lb=0, ub=par['PSmax'])             for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a


xD   = {(t, s, c, a): model.addVar(name=f"xD_{t}_{c}_{a}", lb=0, ub=1)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PGD  = {(t, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0, ub=par['MaxGD'])         for t in Ωt for c in Ωc for a in Ωa}

PAEc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0)             for t in Ωt for c in Ωc for a in Ωa}
PAEd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0)             for t in Ωt for c in Ωc for a in Ωa}
EAE  = {(t, c, a): model.addVar(name=f"EAE_{t}_{c}_{a}", lb=0)              for t in Ωt for c in Ωc for a in Ωa}

γAEc = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"AECharge{a}")      for t in Ωt for c in Ωc for a in Ωa} 
γAEd = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"AEDischarge{a}")   for t in Ωt for c in Ωc for a in Ωa} 

PEVc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0)             for t in Ωt for c in Ωc for a in Ωa}
PEVd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0)             for t in Ωt for c in Ωc for a in Ωa}
SoCEV  = {(t, c, a): model.addVar(name=f"EEV_{t}_{c}_{a}", lb=0)            for t in Ωt for c in Ωc for a in Ωa}


# Define binary decision variables for EV availability scenarios
γEVc = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"EVCharge{a}")    for t in Ωt for c in Ωc for a in Ωa} 
γEVd = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"EVDischarge{a}") for t in Ωt for c in Ωc for a in Ωa} 

# Objective function
model.setObjective(
    par['cIPV'] * PPVmax + par['cIT'] * PGDmax + par['cIPA'] * PAEmax + par['cIEA'] * EAEmax +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['cOS'][t-1] * PSp[t, s, c, a]    for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['cOT'] * PGD[t, c, a]            for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['cCC'] * par['MaxL'] * fp[s]["load"][t-1] * xD[t, s, c, a] for t in Ωt for s in Ωs for s in Ωs for c in Ωc for a in Ωa),
    GRB.MINIMIZE
)

# Assuming Ωt is the list of time intervals
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            if t < Ωa[a]['arrival'][0]:
                model.addConstr(SoCEV[t, c, a] == 0, name=f"EV_SoC_before_{t}_{c}_{a}")
                model.addConstr(PEVc[t, c, a] == 0, name=f"EV_Charge_before_{t}_{c}_{a}")
                model.addConstr(PEVd[t, c, a] == 0, name=f"EV_Discharge_before_{t}_{c}_{a}")
            elif t > Ωa[a]['departure'][-1]:
                model.addConstr(SoCEV[t, c, a] == 0, name=f"EV_SoC_after_{t}_{c}_{a}")
                model.addConstr(PEVc[t, c, a] == 0, name=f"EV_Charge_after_{t}_{c}_{a}")
                model.addConstr(PEVd[t, c, a] == 0, name=f"EV_Discharge_after_{t}_{c}_{a}")
            for n in range(len(Ωa[a]['arrival'])):
                if t == Ωa[a]['departure'][n]:
                    model.addConstr(SoCEV[t, c, a] == 1, name=f"EV_SoC_end_{t}_{c}_{a}")
                if t > Ωa[a]['arrival'][n] and t <= Ωa[a]['departure'][n]:
                    model.addConstr(SoCEV[t, c, a] == SoCEV[t-1,c,a] + Δt * (PEVc[t,c,a] - PEVd[t,c,a])/Ωa[a]['Emax'][n], name=f"EV_SoC_{t}_{c}_{a}")
                    model.addConstr(SoCEV[t, c, a] <= 1, name=f"EV_SoC_max_{t}_{c}_{a}")
                elif t == Ωa[a]['arrival'][n]:
                    model.addConstr(SoCEV[t, c, a] == Ωa[a]['SoCini'][n], name=f"EV_SoC_ini_{t}_{c}_{a}")
                if n < len(Ωa[a]['arrival']) - 1:
                    if t > Ωa[a]['departure'][n] and t < Ωa[a]['arrival'][n+1]:
                        print(t)
                        model.addConstr(SoCEV[t, c, a] == 0, name=f"EV_SoC_between_{t}_{c}_{a}")
                        model.addConstr(PEVc[t, c, a] == 0, name=f"EV_Charge_between_{t}_{c}_{a}")
                        model.addConstr(PEVd[t, c, a] == 0, name=f"EV_Discharge_between_{t}_{c}_{a}")

            model.addConstr(PEVc[t,c,a] <= par['EVPmaxc'] * γEVc[t,c,a], name=f"EV_ChargeMax_{t}_{c}_{a}")
            model.addConstr(PEVd[t,c,a] <= par['EVPmaxd'] * γEVd[t,c,a], name=f"EV_DischargeMax_{t}_{c}_{a}")
            model.addConstr(γEVc[t,c,a] + γEVd[t,c,a] <= 1, name=f"EV_Charge_Discharge_condition_{t}_{c}_{a}")


# Active power balance constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PS[t, s, c, a] + PGD[t, c, a] + fp[s]['pv'][t-1] * PPVmax + PAEd[t, c, a] + PEVd[t, c, a] ==
                    par['MaxL']*fp[s]['load'][t-1] * (1 - xD[t, s, c, a]) + PAEc[t, c, a] + PEVc[t, c, a],
                    name=f"Active_Power_Balance_{t}_{s}_{c}_{a}"
            )



# Substation capacity constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PS[t, s, c, a] <= PSp[t, s, c, a] - PSn[t, s, c, a],
                    name=f"Substation_transformation_{t}_{s}_{c}_{a}"
            )


# Conventional generator capacity constraint
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            model.addConstr(
                PGD[t, c, a] <= PGDmax,
                name=f"Generator_Capacity_{t}_{c}_{a}"
        )

# Energy storage balance constraint
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            if t > 1:
                model.addConstr(
                    EAE[t, c, a] == EAE[t - 1, c, a] + par['alpha'] * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / par['alpha'] - par['beta'] * Δt * EAE[t, c, a],
                    name=f"Energy_Storage_Balance_{t}_{c}_{a}"
            )
            if t == 1:
                model.addConstr(
                    EAE[t, c, a] == par['EAE0'] + par['alpha'] * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / par['alpha'] - par['beta'] * Δt * EAE[t, c, a],
                    name=f"Initial_Energy_Storage_{t}_{c}"
            )
            model.addConstr(
                EAE[t, c, a] <= EAEmax,
                name=f"Max_Energy_Storage_Capacity_{t}_{c}_{a}"
            )
            model.addConstr(
                PAEc[t, c, a] <= PAEmax * γAEc[t, c, a],
                name=f"Max_Injection_Power_{t}_{c}_{a}"
            )
            model.addConstr(
                PAEd[t, c, a] <= PAEmax * γAEd[t, c, a],
                name=f"Max_Extraction_Power_{t}_{c}_{a}"
            )
            model.addConstr(
                γAEc[t, c, a] + γAEd[t, c, a] <= 1,
                name=f"AE_Chage_Discharge_condition_{t}_{c}_{a}"
            )

# Contingency operation constraint
for c in Ωc:
    for s in Ωs:
        for t in range(c, min(max(Ωt), c + int(par['D'] / Δt)) + 1):
            for a in Ωa:
                if c != 0:
                    model.addConstr(
                        PS[t, s, c, a] == 0,
                        name=f"Contingency_Operation_{t}_{s}_{c}_{a}"
                    )

# Solve the model
model.optimize()


print(PPVmax)
print(PGDmax)
print(PAEmax)
print(EAEmax)



# Extract the values for plotting
PS_values   = {(t, s, c, a): PS[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PGD_values  = {(t, c, a): PGD[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
xD_values   = {(t, s, c, a): xD[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PEVc_values = {(t, c, a): PEVc[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
PEVd_values = {(t, c, a): PEVd[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
SoCEV_values  = {(t, c, a): SoCEV[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}


PAEc_values = {(t, c, a): PAEc[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
PAEd_values = {(t, c, a): PAEd[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
EAE_values  = {(t, c, a): EAE[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}

PPVmax_value = PPVmax.x
PGDmax_value = PGDmax.x
PAEmax_value = PAEmax.x
EAEmax_value = EAEmax.x



import matplotlib.pyplot as plt


num_c = len(Ωc)
num_a = len(Ωa)

from itertools import product


Ωc_a_pairs = list(product(Ωc, Ωa))

Ωc_a_pairs1 = list(product(Ωc, Ωa))

# Create a 2x2 grid of subplots

fig, axes2 = plt.subplots(2, 2, figsize=(15, 10))
# Create a 2x2 grid of subplots dynamically based on the number of combinations
num_subplots = len(Ωc_a_pairs)
num_cols = 2
num_rows = -(-num_subplots // num_cols)  # Ceiling division to determine the number of rows

# Create subplots
fig, axes1 = plt.subplots(num_rows, num_cols, figsize=(15, 10))
axes1 = axes1.flatten()
axes2 = axes2.flatten()

# Loop over combinations of c and a
for idx, (c, a) in enumerate(Ωc_a_pairs):  # Using enumerate to loop over both index and value
    # Power-related plots
    axes1[idx].plot(Ωt, [PS_values[t, s, c, a] for t in Ωt], label="PS")
    axes1[idx].plot(Ωt, [PGD_values[t, c, a] for t in Ωt], label="PGD")
    axes1[idx].plot(Ωt, [fp[s]['load'][t-1]*par['MaxL'] for t in Ωt], label="Demand")
    axes1[idx].plot(Ωt, [PAEc_values[t, c, a] for t in Ωt], label="PAEc")
    axes1[idx].plot(Ωt, [-1 * PAEd_values[t, c, a] for t in Ωt], label="PAEd")
    axes1[idx].plot(Ωt, [PEVc_values[t, c, a] for t in Ωt], label="PEVc")
    axes1[idx].plot(Ωt, [-1 * PEVd_values[t, c, a] for t in Ωt], label="PEVd")
    axes1[idx].legend()
    axes1[idx].set_xlabel("Timestamp")
    axes1[idx].set_ylabel("Power [kW]")
    axes1[idx].set_title(f"Power Values - c{c}, a{a}")


for idx, (c, a) in enumerate(Ωc_a_pairs1):  # Assuming Ωc_a_pairs is a list of pairs (c, a)
    axes2[idx].plot(Ωt, [EAE_values[t, c, a]/EAEmax_value for t in Ωt], label="EAE")
    axes2[idx].plot(Ωt, [SoCEV_values[t, c, a] for t in Ωt], label="SoCEV")
    axes2[idx].legend()
    axes2[idx].set_xlabel("Timestamp")
    axes2[idx].set_ylabel("Power [kW]")
    axes2[idx].set_title(f"EV Power - c{c}, a{a}")


# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()