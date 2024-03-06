from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import time
import json
import os

fp = json.load(open('parameters/scenarios15m.json', 'r'))
par = json.load(open('parameters/parameters.json', 'r'))
contingency = json.load(open('parameters/contingency.json', 'r'))

Ωa = json.load(open('parameters/EV.json', 'r'))
Ωt = list(range(1, 97))
Ωc = contingency['timestamp']
Ωs = fp.keys() # Load and 
Δt = 0.25  # Define the time interval in hours

πc = {contingency['timestamp'][i]: contingency['probability'][i]  for i in range(0,len(Ωc))}  # Define the probability of each contingency
πs = {s: fp[s]['prob'] for s in Ωs}  # Define the probability of each scenario

start = time.time()

# Create a Gurobi model
model = gp.Model(par['IonLitResNCAInd']['model name'])

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0)
PGDmax = model.addVar(name="PGDmax", lb=0)
PAEmax = model.addVar(name="PAEmax", lb=0)
EAEmax = model.addVar(name="EAEmax", lb=0)

OPEX = model.addVar(name="OPEX", lb=0)
CAPEX = model.addVar(name="CAPEX", lb=0)
OPEX_yearly = model.addVar(name="OPEX", lb=0)


PS   = {(t, s, c, a): model.addVar(name=f"PS_{t}_{s}_{c}_{a}", lb=par['IonLitResNCAInd']['PSmin'], ub=par['IonLitResNCAInd']['PSmax'])    for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSp   = {(t, s, c, a): model.addVar(name=f"PSp_{t}_{s}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSn   = {(t, s, c, a): model.addVar(name=f"PSn_{t}_{s}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a


xD   = {(t, s, c, a): model.addVar(name=f"xD_{t}_{s}_{c}_{a}", lb=0, ub=1)              for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PGD  = {(t, s, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0, ub=par['IonLitResNCAInd']['MaxGD'])     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PAEc = {(t, s, c, a): model.addVar(name=f"PAEi_{t}_{s}_{c}_{a}", lb=0)                     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PAEd = {(t, s, c, a): model.addVar(name=f"PAEe_{t}_{s}_{c}_{a}", lb=0)                     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
EAE  = {(t, s, c, a): model.addVar(name=f"EAE_{t}_{s}_{c}_{a}", lb=0)                      for t in Ωt for s in Ωs for c in Ωc for a in Ωa}



PEVc = {(t, s, c, a): model.addVar(name=f"PAEi_{t}_{s}_{c}_{a}", lb=0, ub=par['IonLitResNCAInd']['EVPmaxc'])  for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PEVd = {(t, s, c, a): model.addVar(name=f"PAEe_{t}_{s}_{c}_{a}", lb=0, ub=par['IonLitResNCAInd']['EVPmaxd'])  for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
SoCEV  = {(t, s, c, a): model.addVar(name=f"EEV_{t}_{s}_{c}_{a}", lb=0)                    for t in Ωt for s in Ωs for c in Ωc for a in Ωa}


 

# Objective function
model.setObjective(OPEX + CAPEX, GRB.MINIMIZE)

# Constraints
model.addConstr(OPEX_yearly == 
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['IonLitResNCAInd']['cOS'][t-1] * PSp[t, s, c, a]    for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * (par['IonLitResNCAInd']['cOT'] + par['IonLitResNCAInd']['GDM']) * PGD[t, s, c, a]            for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['IonLitResNCAInd']['EAM'] * PAEd[t, s, c, a]            for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['IonLitResNCAInd']['cCC'] * par['IonLitResNCAInd']['MaxL'] * fp[s]["load"][t-1] * xD[t, s, c, a] for t in Ωt for s in Ωs for s in Ωs for c in Ωc for a in Ωa),
    name="OPEX_yearly"
)

model.addConstr(CAPEX == par['IonLitResNCAInd']['cIPV'] * PPVmax + par['IonLitResNCAInd']['cIT'] * PGDmax + par['IonLitResNCAInd']['cIPA'] * PAEmax + par['IonLitResNCAInd']['cIEA'] * EAEmax, name="CAPEX")
model.addConstr(OPEX == gp.quicksum( (1/(1+par['IonLitResNCAInd']['rate'])**y) * OPEX_yearly for y in range(1,par['IonLitResNCAInd']['nyears']+1)))

# Assuming Ωt is the list of time intervals
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                if t < Ωa[a]['arrival'][0]:
                    model.addConstr(SoCEV[t, s, c, a] == 0, name=f"EV_SoC_before_{t}_{c}_{a}")
                    model.addConstr(PEVc[t, s, c, a] == 0, name=f"EV_Charge_before_{t}_{c}_{a}")
                    model.addConstr(PEVd[t, s, c, a] == 0, name=f"EV_Discharge_before_{t}_{c}_{a}")
                elif t > Ωa[a]['departure'][-1]:
                    model.addConstr(SoCEV[t, s, c, a] == 0, name=f"EV_SoC_after_{t}_{c}_{a}")
                    model.addConstr(PEVc[t, s, c, a] == 0, name=f"EV_Charge_after_{t}_{c}_{a}")
                    model.addConstr(PEVd[t, s, c, a] == 0, name=f"EV_Discharge_after_{t}_{c}_{a}")
                for n in range(len(Ωa[a]['arrival'])):
                    if t == Ωa[a]['departure'][n]:
                        model.addConstr(SoCEV[t, s, c, a] == 1, name=f"EV_SoC_end_{t}_{c}_{a}")
                    if t > Ωa[a]['arrival'][n] and t <= Ωa[a]['departure'][n]:
                        model.addConstr(SoCEV[t, s, c, a] == SoCEV[t-1, s, c, a] + Δt * (PEVc[t, s, c, a] - PEVd[t, s, c, a])/Ωa[a]['Emax'][n], name=f"EV_SoC_{t}_{c}_{a}")
                        model.addConstr(SoCEV[t, s, c, a] <= 1, name=f"EV_SoC_max_{t}_{c}_{a}")
                        model.addConstr(PEVc[t, s, c, a] <= (1 - SoCEV[t-1, s, c, a]) * Ωa[a]['Emax'][n]/Δt, name=f"EV_Charge_Constraint_{t}_{c}_{a}")
                        model.addConstr(PEVd[t, s, c, a] <= SoCEV[t-1, s, c, a] * Ωa[a]['Emax'][n]/Δt, name=f"EV_Discharge_Constraint_{t}_{c}_{a}")
                        model.addConstr(PEVd[t, s, c, a] <= par['IonLitResNCAInd']['EVPmaxd'] - (par['IonLitResNCAInd']['EVPmaxd']/par['IonLitResNCAInd']['EVPmaxc'])*PEVc[t, s, c, a], name=f"EV_Discharge_Max_{t}_{c}_{a}")
                    
                    elif t == Ωa[a]['arrival'][n]:
                        model.addConstr(SoCEV[t, s, c, a] == Ωa[a]['SoCini'][n], name=f"EV_SoC_ini_{t}_{c}_{a}")
                    elif n < len(Ωa[a]['arrival']) - 1:
                        if t > Ωa[a]['departure'][n] and t < Ωa[a]['arrival'][n+1]:
                            model.addConstr(SoCEV[t, s, c, a] == 0, name=f"EV_SoC_between_{t}_{c}_{a}")
                            model.addConstr(PEVc[t, s, c, a] == 0, name=f"EV_Charge_between_{t}_{c}_{a}")
                            model.addConstr(PEVd[t, s, c, a] == 0, name=f"EV_Discharge_between_{t}_{c}_{a}")

                # model.addConstr(PEVc[t,c,a] <= par['IonLitResNCAInd']['EVPmaxc'], name=f"EV_ChargeMax_{t}_{c}_{a}")
                # model.addConstr(PEVd[t,c,a] <= par['IonLitResNCAInd']['EVPmaxd'], name=f"EV_DischargeMax_{t}_{c}_{a}")
    


# Active power balance constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PS[t, s, c, a] + PGD[t, s, c, a] + fp[s]['pv'][t-1] * PPVmax + PAEd[t, s, c, a] + PEVd[t, s, c, a] ==
                    par['IonLitResNCAInd']['MaxL']*fp[s]['load'][t-1] * (1 - xD[t, s, c, a]) + PAEc[t, s, c, a] + PEVc[t, s, c, a],
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
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PGD[t, s, c, a] <= PGDmax,
                    name=f"Generator_Capacity_{t}_{s}_{c}_{a}"
        )

# Energy storage balance constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                if t > 1:
                    model.addConstr(
                        EAE[t, s, c, a] == EAE[t - 1, s, c, a] + par['IonLitResNCAInd']['alpha'] * Δt * PAEc[t, s, c, a] - Δt * PAEd[t, s, c, a] / par['IonLitResNCAInd']['alpha'] - EAE[t, s, c, a] * par['IonLitResNCAInd']['beta'],
                        name=f"Energy_Storage_Balance_{t}_{s}_{c}_{a}"
                    )
                    model.addConstr(PAEc[t, s, c, a] <= (EAEmax - EAE[t-1, s, c, a])/(par['IonLitResNCAInd']['alpha'] * Δt), name=f"Max_Charge_BESS_1_{t}_{s}_{c}_{a}")
                    model.addConstr(PAEd[t, s, c, a] <= EAE[t-1, s, c, a]*par['IonLitResNCAInd']['alpha']/Δt, name=f"Max_Discharge_BESS_1_{t}_{s}_{c}_{a}")
                    
                if t == 1:
                    model.addConstr(
                        EAE[t, s, c, a] == par['IonLitResNCAInd']['EAE0'] * EAEmax + par['IonLitResNCAInd']['alpha'] * Δt * PAEc[t, s, c, a] - Δt * PAEd[t, s, c, a] / par['IonLitResNCAInd']['alpha'],
                        name=f"Initial_Energy_Storage_initial_{t}_{s}_{c}_{a}"
                    )
                    model.addConstr(PAEc[t, s, c, a] <= (EAEmax - par['IonLitResNCAInd']['EAE0'])/(par['IonLitResNCAInd']['alpha'] * Δt), name=f"Max_Charge_BESS_2_{t}_{s}_{c}_{a}")
                    model.addConstr(PAEd[t, s, c, a] <= par['IonLitResNCAInd']['EAE0']*par['IonLitResNCAInd']['alpha']/Δt, name=f"Max_Discharge_BESS_2_{t}_{s}_{c}_{a}")

                model.addConstr(PAEd[t, s, c, a] <= PAEmax - PAEc[t, s, c, a], name=f"Max_Discharge_BESS_General_{t}_{s}_{c}_{a}")
                model.addConstr(EAE[t, s, c, a] <= EAEmax, name=f"Max_Energy_Storage_Capacity_{t}_{s}_{c}_{a}")
                model.addConstr(PAEc[t, s, c, a] <= PAEmax, name=f"Max_Injection_Power_{t}_{s}_{c}_{a}")
                model.addConstr(PAEd[t, s, c, a] <= PAEmax, name=f"Max_Extraction_Power_{t}_{s}_{c}_{a}")
                model.addConstr(PAEmax <= EAEmax, name=f"Max_Injection_Power_{t}_{s}_{c}_{a}")


           

# Contingency operation constraint
for c in Ωc:
    for s in Ωs:
        for t in range(c, min(max(Ωt), c + int(par['IonLitResNCAInd']['D'] / Δt)) + 1):
            for a in Ωa:
                if c != 0:
                    model.addConstr(
                        PS[t, s, c, a] == 0,
                        name=f"Contingency_Operation_{t}_{s}_{c}_{a}"
                    )

for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                if t < c:
                    model.addConstr(EAE[t, s, c, a] >= 0.5*EAEmax, name=f"BESS_before_Contingency_Operation_{t}_{s}_{c}_{a}")
                    model.addConstr(EAE[t, s, c, a] == EAE[t, s, 0, a], name=f"BESS_Contingency_Operation_{t}_{s}_{c}_{a}")
                    model.addConstr(SoCEV[t, s, c, a] == SoCEV[t, s, 0, a], name=f"EV_before_Contingency_Operation_{t}_{s}_{c}_{a}")

# Solve the model
model.optimize()

end = time.time()

print(PPVmax)
print(PGDmax)
print(PAEmax)
print(EAEmax)
print(f'total execution time (LP): {end - start} seconds')

for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                if (not(PAEc[t, s, c, a] == 0) and not(PAEd[t, s, c, a] == 0)):
                    print("DEU ERRO")

# Extract the values for plotting
PS_values   = {(t, s, c, a): PS[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PGD_values  = {(t, s, c, a): PGD[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
xD_values   = {(t, s, c, a): xD[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PEVc_values = {(t, s, c, a): PEVc[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PEVd_values = {(t, s, c, a): PEVd[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
SoCEV_values  = {(t, s, c, a): SoCEV[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}


PAEc_values = {(t, s, c, a): PAEc[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PAEd_values = {(t, s, c, a): PAEd[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
EAE_values  = {(t, s, c, a): EAE[t, s, c, a].x for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PPVmax_value = PPVmax.x
PGDmax_value = PGDmax.x
PAEmax_value = PAEmax.x
EAEmax_value = EAEmax.x


# if not os.path.exists("Results"):
#     os.makedirs("Results")


import matplotlib.pyplot as plt

# Create subplots with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Convert Ωs keys to list and then slice
Ωs_list = list(Ωs)
for idx, s in enumerate(Ωs_list[:2]):
    ax = axs[idx]
    ax.set_title(f"Scenario {idx+1}")  # Set subplot title
    for c in Ωc:
        for a in list(Ωa.keys())[:2]:  # Extract keys of Ωa as a list and then slice
            ax.plot(Ωt, [PS_values[t, s, c, a] for t in Ωt], label="EDS")
            ax.plot(Ωt, [PGD_values[t, s, c, a] for t in Ωt], label="Thermal Generator")
            ax.plot(Ωt, [fp[s]['load'][t-1]*par['IonLitResNCAInd']['MaxL']*(1 - xD_values[t,s,c,a]) for t in Ωt], label="Demand")
            ax.plot(Ωt, [-1*fp[s]['pv'][t-1] * PPVmax_value  for t in Ωt], label="PV")
            if EAEmax_value > 0:
                ax.plot(Ωt, [PAEc_values[t, s, c, a] - PAEd_values[t, s, c, a] for t in Ωt], label="BESS")
            ax.plot(Ωt, [PEVc_values[t, s, c, a] - PEVd_values[t, s, c, a] for t in Ωt], label="EV")

# Add legend to the last subplot
axs[-1].legend()
plt.xlabel("Timestamp")
plt.ylabel("Power [kW]")
plt.show()

# Plot for two values in Ωa
plt.figure()
for s in Ωs:
    for c in Ωc:
        for idx, a in enumerate(list(Ωa.keys())[:2]):  # Extract keys of Ωa as a list and then slice
            plt.plot(Ωt, [SoCEV_values[t, s, c, a] for t in Ωt], label=f"Scenario {idx+1}")

plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Power [kW]")
plt.show()

# Plot for par['IonLitResNCAInd']['cOS']
plt.figure()
plt.plot(Ωt, [par['IonLitResNCAInd']['cOS'] for _ in Ωt], label="cOS")

plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Power [kW]")
plt.show()
