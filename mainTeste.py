from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import json

fPV = json.load(open('parameters/PV.json', 'r'))
fPD = json.load(open('parameters/PD.json', 'r'))

Ωs = ["1"]
Ωt = list(range(1, 97))
Ωc = [0, 72, 80, 88]
Ωa = json.load(open('parameters/EV.json', 'r'))

Δt = 0.25  # Define the time interval in hours

# Create a Gurobi model
model = gp.Model("Microgrid Optimization")

PD = json.load(open('parameters/PD.json', 'r'))

fPV = json.load(open('parameters/PV.json','r'))

cOS = json.load(open('parameters/cOS.json','r'))
# Create cOS dictionary

p = {0: 0.985, 72: 0.005, 80: 0.005, 88: 0.005}


PSmax = 1000
IPPVmax = 400
cIPV = 900
cCC = 1500
D = 2
cIT = 2000
cOT = 1.280
IPGDmax = 100
cIPA = 2000
cIEA = 2000
EAE0 = 0
alpha = 0.95
beta = 0.05


# Define maximum charging and discharging power EV
MaxChargePower = 50  # Adjust this value based on your requirements
MaxDischargePower = 0  # Adjust this value based on your requirements

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0, ub=IPPVmax)
PGDmax = model.addVar(name="PGDmax", lb=0, ub=IPGDmax)
PAEmax = model.addVar(name="PAEmax", lb=0) #Both this and bottom one could be selected from a pool
EAEmax = model.addVar(name="EAEmax", lb=0)

PS   = {(t, c, a): model.addVar(name=f"PS_{t}_{c}_{a}", lb=0)       for t in Ωt for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PGD  = {(t, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0)      for t in Ωt for c in Ωc for a in Ωa}
xD   = {(t, c, a): model.addVar(name=f"xD_{t}_{c}_{a}", lb=0, ub=1) for t in Ωt for c in Ωc for a in Ωa}

PAEc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0)     for t in Ωt for c in Ωc for a in Ωa}
PAEd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0)     for t in Ωt for c in Ωc for a in Ωa}
EAE  = {(t, c, a): model.addVar(name=f"EAE_{t}_{c}_{a}", lb=0)      for t in Ωt for c in Ωc for a in Ωa}

γAEc = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"AECharge{a}")    for t in Ωt for c in Ωc for a in Ωa} 
γAEd = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"AEDischarge{a}") for t in Ωt for c in Ωc for a in Ωa} 

PEVc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0)     for t in Ωt for c in Ωc for a in Ωa}
PEVd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0)     for t in Ωt for c in Ωc for a in Ωa}
SoCEV  = {(t, c, a): model.addVar(name=f"EEV_{t}_{c}_{a}", lb=0)      for t in Ωt for c in Ωc for a in Ωa}


# Define binary decision variables for EV availability scenarios
γEVc = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"EVCharge{a}")    for t in Ωt for c in Ωc for a in Ωa} 
γEVd = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"EVDischarge{a}") for t in Ωt for c in Ωc for a in Ωa} 

# Objective function
model.setObjective(
    cIPV * PPVmax + cIT * PGDmax + cIPA * PAEmax + cIEA * EAEmax +
    365 * gp.quicksum(p[c] * Δt * cOS['1'][t - 1] * PS[t, c, a] for t in Ωt for c in Ωc for a in Ωa) +
    365 * gp.quicksum(p[c] * Δt * cOT * PGD[t, c, a] for t in Ωt for c in Ωc for a in Ωa) +
    365 * gp.quicksum(p[c] * Δt * cCC * PD['1'][t-1] * xD[t, c, a] for t in Ωt for c in Ωc for a in Ωa),
    GRB.MINIMIZE
)

# Assuming Ωt is the list of time intervals
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            if t < Ωa[a]['arrival'][0]:
                model.addConstr(SoCEV[t,c,a] == 0)
                model.addConstr(PEVc[t,c,a] == 0)
                model.addConstr(PEVd[t,c,a] == 0)
            elif t > Ωa[a]['departure'][-1]:
                model.addConstr(SoCEV[t,c,a] == 0)
                model.addConstr(PEVc[t,c,a] == 0)
                model.addConstr(PEVd[t,c,a] == 0)
            for n in range(len(Ωa[a]['arrival'])):
                if t > Ωa[a]['arrival'][n] and t < Ωa[a]['departure'][n]:
                    model.addConstr(SoCEV[t,c,a] == SoCEV[t-1,c,a] + Δt * (PEVc[t,c,a] - PEVd[t,c,a])/Ωa[a]['Emax'][n])
                    model.addConstr(SoCEV[t,c,a] <= 1)
                elif t == Ωa[a]['arrival'][n]:
                    model.addConstr(SoCEV[t,c,a] == Ωa[a]['SoCini'][n])
                elif t == Ωa[a]['departure'][n]:
                    model.addConstr(SoCEV[t,c,a] == 1)
                elif n < len(Ωa[a]['arrival']) - 1:
                    if t > Ωa[a]['departure'][n] and t < Ωa[a]['arrival'][n+1]:
                        model.addConstr(SoCEV[t,c,a] == 0)
                        model.addConstr(PEVc[t,c,a] == 0)
                        model.addConstr(PEVd[t,c,a] == 0)

            model.addConstr(PEVc[t,c,a] <= MaxChargePower * γEVc[t,c,a])
            model.addConstr(PEVd[t,c,a] <= MaxDischargePower * γEVd[t,c,a])
            model.addConstr(γEVc[t,c,a] + γEVd[t,c,a] <= 1)



# Active power balance constraint
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            model.addConstr(
                PS[t, c, a] + PGD[t, c, a] + fPV['1'][t-1] * PPVmax + PAEd[t, c, a] + PEVd[t, c, a] ==
                PD['1'][t-1] * (1 - xD[t, c, a]) + PAEc[t, c, a] + PEVc[t, c, a],
                name=f"Active_Power_Balance_{t}_{c}_{a}"
        )



# Substation capacity constraint
for t in Ωt:
    for c in Ωc:
        for a in Ωa:
            model.addConstr(
                PS[t, c, a] <= PSmax,
                name=f"Substation_Capacity_{t}_{c}_{a}"
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
                    EAE[t, c, a] == EAE[t - 1, c, a] + alpha * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / alpha - beta * Δt * EAE[t, c, a],
                    name=f"Energy_Storage_Balance_{t}_{c}_{a}"
            )
            if t == 1:
                model.addConstr(
                    EAE[t, c, a] == EAE0 + alpha * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / alpha - beta * Δt * EAE[t, c, a],
                    name=f"Initial_Energy_Storage_{t}_{c}"
            )
            model.addConstr(
                EAE[t, c, a] <= EAEmax,
                name=f"Max_Energy_Storage_Capacity_{t}_{c}_{a}"
            )
            model.addConstr(
                PAEc[t, c, a] <= PAEmax * γAEc[t, c, a],
                name=f"Max_Injection_Power_{t}_{c}+{a}"
            )
            model.addConstr(
                PAEd[t, c, a] <= PAEmax * γAEd[t, c, a],
                name=f"Max_Extraction_Power_{t}_{c}+{a}"
            )
            model.addConstr(
                γAEc[t, c, a] + γAEd[t, c, a] <= 1,
                name=f"AE_Chage_Discharge_condition_{t}_{c}+{a}"
            )

# Contingency operation constraint
for c in Ωc:
    for t in range(c, min(max(Ωt), c + int(D / Δt)) + 1):
        for a in Ωa:
            if c != 0:
                model.addConstr(
                    PS[t, c, a] == 0,
                    name=f"Contingency_Operation_{c}_{t}_{a}"
                )

# Solve the model
model.optimize()




print(PPVmax)
print(PGDmax)
print(PAEmax)
print(EAEmax)


# Extract the values for plotting
PS_values   = {(t, c, a): PS[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}
PGD_values  = {(t, c, a): PGD[t, c, a].x for t in Ωt for c in Ωc for a in Ωa}

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
fig, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig, axes2 = plt.subplots(2, 2, figsize=(15, 10))
# Flatten the 2D array of subplots for easy indexing
axes1 = axes1.flatten()
axes2 = axes2.flatten()

# Loop over combinations of c and a
for idx, (c, a) in enumerate(Ωc_a_pairs):  # Assuming Ωc_a_pairs is a list of pairs (c, a)
    # Power-related plots
    axes1[idx].plot(Ωt, [PS_values[t, c, a] for t in Ωt], label="PS")
    axes1[idx].plot(Ωt, [PGD_values[t, c, a] for t in Ωt], label="PGD")
    axes1[idx].plot(Ωt, [PD['1'][t-1] for t in Ωt], label="PD")
    axes1[idx].plot(Ωt, [fPV['1'][t-1] * PPVmax_value for t in Ωt], label="fPV")
    axes1[idx].plot(Ωt, [PAEc_values[t, c, a] for t in Ωt], label="PAEc")
    axes1[idx].plot(Ωt, [-1 * PAEd_values[t, c, a] for t in Ωt], label="PAEd")
    axes1[idx].plot(Ωt, [PAEc_values[t, c, a] for t in Ωt], label="PEVc")
    axes1[idx].plot(Ωt, [-1 * PAEd_values[t, c, a] for t in Ωt], label="PEVd")
    axes1[idx].legend()
    axes1[idx].set_xlabel("Timestamp")
    axes1[idx].set_ylabel("Power [kW]")
    axes1[idx].set_title(f"Power Values - c{c}, a{a}")
print(cOS)
for idx, (c, a) in enumerate(Ωc_a_pairs1):  # Assuming Ωc_a_pairs is a list of pairs (c, a)
    axes2[idx].plot(Ωt, [EAE_values[t, c, a]/EAEmax_value for t in Ωt], label="EAE")
    axes2[idx].plot(Ωt, [SoCEV_values[t, c, a] for t in Ωt], label="SoCEV")
    axes2[idx].plot(Ωt, cOS['1'], label="cOS")  # Plot cOS values
    axes2[idx].legend()
    axes2[idx].set_xlabel("Timestamp")
    axes2[idx].set_ylabel("Power [kW]")
    axes2[idx].set_title(f"EV Power - c{c}, a{a}")


# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
