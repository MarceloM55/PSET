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
Ωs = fp.keys()
Δt = 0.25  # Define the time interval in hours

πc = {contingency['timestamp'][i]: contingency['probability'][i]  for i in range(0,len(Ωc))}  # Define the probability of each contingency
πs = {s: fp[s]['prob'] for s in Ωs}  # Define the probability of each scenario

start = time.time()

# Create a Gurobi model
model = gp.Model(par['baseModel']['model name'])

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0)
PGDmax = model.addVar(name="PGDmax", lb=0)
PAEmax = model.addVar(name="PAEmax", lb=0)
EAEmax = model.addVar(name="EAEmax", lb=0)

OPEX = model.addVar(name="OPEX", lb=0)
CAPEX = model.addVar(name="CAPEX", lb=0)
OPEX_yearly = model.addVar(name="OPEX", lb=0)


PS   = {(t, s, c, a): model.addVar(name=f"PS_{t}_{c}_{a}", lb=par['baseModel']['PSmin'], ub=par['baseModel']['PSmax'])    for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSp   = {(t, s, c, a): model.addVar(name=f"PSp_{t}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSn   = {(t, s, c, a): model.addVar(name=f"PSn_{t}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a


xD   = {(t, s, c, a): model.addVar(name=f"xD_{t}_{c}_{a}", lb=0, ub=1)              for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PGD  = {(t, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0, ub=par['baseModel']['MaxGD'])     for t in Ωt for c in Ωc for a in Ωa}

PAEc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0)                     for t in Ωt for c in Ωc for a in Ωa}
PAEd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0)                     for t in Ωt for c in Ωc for a in Ωa}
EAE  = {(t, c, a): model.addVar(name=f"EAE_{t}_{c}_{a}", lb=0)                      for t in Ωt for c in Ωc for a in Ωa}



PEVc = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0, ub=par['baseModel']['EVPmaxc'])  for t in Ωt for c in Ωc for a in Ωa}
PEVd = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0, ub=par['baseModel']['EVPmaxd'])  for t in Ωt for c in Ωc for a in Ωa}
SoCEV  = {(t, c, a): model.addVar(name=f"EEV_{t}_{c}_{a}", lb=0)                    for t in Ωt for c in Ωc for a in Ωa}


 

# Objective function
model.setObjective(OPEX + CAPEX, GRB.MINIMIZE)

# Constraints
model.addConstr(OPEX_yearly == 
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['baseModel']['cOS'][t-1] * PSp[t, s, c, a]    for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['baseModel']['cOT'] * PGD[t, c, a]            for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * Δt * par['baseModel']['cCC'] * par['baseModel']['MaxL'] * fp[s]["load"][t-1] * xD[t, s, c, a] for t in Ωt for s in Ωs for s in Ωs for c in Ωc for a in Ωa),
    name="OPEX_yearly"
)

model.addConstr(CAPEX == par['baseModel']['cIPV'] * PPVmax + par['baseModel']['cIT'] * PGDmax + par['baseModel']['cIPA'] * PAEmax + par['baseModel']['cIEA'] * EAEmax, name="CAPEX")
model.addConstr(OPEX == gp.quicksum( (1/(1+par['baseModel']['rate'])**y) * OPEX_yearly for y in range(1,par['baseModel']['nyears']+1)))

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
                    model.addConstr(PEVc[t,c,a] <= (1 - SoCEV[t-1, c, a]) * Ωa[a]['Emax'][n]/Δt, name=f"EV_Charge_Constraint_{t}_{c}_{a}")
                    model.addConstr(PEVd[t,c,a] <= SoCEV[t-1, c, a] * Ωa[a]['Emax'][n]/Δt, name=f"EV_Discharge_Constraint_{t}_{c}_{a}")
                    model.addConstr(PEVd[t,c,a] <= par['baseModel']['EVPmaxd'] - (par['baseModel']['EVPmaxd']/par['baseModel']['EVPmaxc'])*PEVc[t,c,a], name=f"EV_Discharge_Max_{t}_{c}_{a}")
                
                elif t == Ωa[a]['arrival'][n]:
                    model.addConstr(SoCEV[t, c, a] == Ωa[a]['SoCini'][n], name=f"EV_SoC_ini_{t}_{c}_{a}")
                elif n < len(Ωa[a]['arrival']) - 1:
                    if t > Ωa[a]['departure'][n] and t < Ωa[a]['arrival'][n+1]:
                        model.addConstr(SoCEV[t, c, a] == 0, name=f"EV_SoC_between_{t}_{c}_{a}")
                        model.addConstr(PEVc[t, c, a] == 0, name=f"EV_Charge_between_{t}_{c}_{a}")
                        model.addConstr(PEVd[t, c, a] == 0, name=f"EV_Discharge_between_{t}_{c}_{a}")

            # model.addConstr(PEVc[t,c,a] <= par['baseModel']['EVPmaxc'], name=f"EV_ChargeMax_{t}_{c}_{a}")
            # model.addConstr(PEVd[t,c,a] <= par['baseModel']['EVPmaxd'], name=f"EV_DischargeMax_{t}_{c}_{a}")
  


# Active power balance constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PS[t, s, c, a] + PGD[t, c, a] + fp[s]['pv'][t-1] * PPVmax + PAEd[t, c, a] + PEVd[t, c, a] ==
                    par['baseModel']['MaxL']*fp[s]['load'][t-1] * (1 - xD[t, s, c, a]) + PAEc[t, c, a] + PEVc[t, c, a],
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
                    EAE[t, c, a] == EAE[t - 1, c, a] + par['baseModel']['alpha'] * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / par['baseModel']['alpha'],
                    name=f"Energy_Storage_Balance_{t}_{c}_{a}"
                )
                model.addConstr(PAEc[t, c, a] <= (EAEmax - EAE[t-1, c, a])/(par['baseModel']['alpha'] * Δt), name=f"Max_Charge_BESS_1_{t}_{c}_{a}")
                model.addConstr(PAEd[t, c, a] <= EAE[t-1, c, a]*par['baseModel']['alpha']/Δt, name=f"Max_Discharge_BESS_1_{t}_{c}_{a}")
                
            if t == 1:
                model.addConstr(
                    EAE[t, c, a] == par['baseModel']['EAE0'] * EAEmax + par['baseModel']['alpha'] * Δt * PAEc[t, c, a] - Δt * PAEd[t, c, a] / par['baseModel']['alpha'],
                    name=f"Initial_Energy_Storage_{t}_{c}"
                )
                model.addConstr(PAEc[t, c, a] <= (EAEmax - par['baseModel']['EAE0'])/(par['baseModel']['alpha'] * Δt), name=f"Max_Charge_BESS_2_{t}_{c}_{a}")
                model.addConstr(PAEd[t, c, a] <= par['baseModel']['EAE0']*par['baseModel']['alpha']/Δt, name=f"Max_Discharge_BESS_2_{t}_{c}_{a}")

            model.addConstr(PAEd[t, c, a] <= PAEmax - PAEc[t, c, a], name=f"Max_Discharge_BESS_General_{t}_{c}_{a}")
            model.addConstr(EAE[t, c, a] <= EAEmax, name=f"Max_Energy_Storage_Capacity_{t}_{c}_{a}")
            model.addConstr(PAEc[t, c, a] <= PAEmax, name=f"Max_Injection_Power_{t}_{c}_{a}")
            model.addConstr(PAEd[t, c, a] <= 0.5 * PAEmax, name=f"Max_Extraction_Power_{t}_{c}_{a}")


           

# Contingency operation constraint
for c in Ωc:
    for s in Ωs:
        for t in range(c, min(max(Ωt), c + int(par['baseModel']['D'] / Δt)) + 1):
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
                    model.addConstr(EAE[t, c, a] >= 0.5*EAEmax, name=f"BESS_before_Contingency_Operation_{t}_{s}_{c}_{a}")
                    model.addConstr(EAE[t, c, a] == EAE[t, 0, a], name=f"BESS_Contingency_Operation_{t}_{s}_{c}_{a}")
                    model.addConstr(SoCEV[t, c, a] == SoCEV[t, 0, a], name=f"EV_before_Contingency_Operation_{t}_{s}_{c}_{a}")


# Solve the model
model.optimize()

end = time.time()

print(PPVmax)
print(PGDmax)
print(PAEmax)
print(EAEmax)

# if not os.path.exists("Results"):
#     os.makedirs("Results")


# for s in Ωs:
#     for c in Ωc:
#         for a in Ωa:
#             plt.figure()
#             plt.plot(Ωt, [PS_values[t, s, c, a] for t in Ωt], label="EDS")
#             plt.plot(Ωt, [PGD_values[t, c, a] for t in Ωt], label="Thermal Generator")
#             plt.plot(Ωt, [fp[s]['load'][t-1]*par['baseModel']['MaxL']*(1 - xD_values[t,s,c,a]) for t in Ωt], label="Demand")
#             plt.plot(Ωt, [-1*fp[s]['pv'][t-1] * PPVmax_value  for t in Ωt], label="PV")
#             if EAEmax_value > 0:
#                 plt.plot(Ωt, [PAEc_values[t, c, a] - PAEd_values[t, c, a] for t in Ωt], label="BESS")
#             plt.plot(Ωt, [PEVc_values[t, c, a] - PEVd_values[t, c, a] for t in Ωt], label="EV")
#             plt.legend()
#             plt.xlabel("Timestamp")
#             plt.ylabel("Power [kW]")
#             plt.savefig(f"Results/operation_s{s}_c{c}_a{a}.png")
#             plt.close()

# for c in Ωc:
#     for a in Ωa:
#         plt.figure()
#         if EAEmax_value > 0:
#             plt.plot(Ωt, [EAE_values[t, c, a]/EAEmax_value for t in Ωt], label="EAE")
#         plt.plot(Ωt, [SoCEV_values[t, c, a] for t in Ωt], label="SoCEV")
#         plt.legend()
#         plt.xlabel("Timestamp")
#         plt.ylabel("SoC")
#         plt.savefig(f"Results/storage_c{c}_a{a}.png")
#         plt.close()



