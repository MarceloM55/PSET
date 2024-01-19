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



PD_data = [158.900000000000006, 158.609473684210542, 158.318947368421050, 158.028421052631586,
        157.737894736842094, 162.394736842105260, 167.793684210526322, 173.192631578947356,
        178.591578947368419, 180.053684210526313, 180.126315789473693, 180.198947368421074,
        180.271578947368425, 180.948421052631574, 182.013684210526321, 183.078947368421041,
        184.144210526315788, 181.735789473684207, 175.537894736842077, 169.340000000000003,
        163.142105263157902, 156.674736842105261, 149.702105263157904, 142.729473684210518,
        135.756842105263161, 128.626315789473693, 120.927368421052634, 113.228421052631575,
        105.529473684210529, 98.655789473684209, 100.447368421052630, 102.238947368421051,
        104.030526315789473, 105.822105263157908, 126.232631578947348, 147.489473684210452,
        168.746315789473726, 190.003157894736830, 195.159999999999997, 196.927368421052620,
        198.694736842105272, 200.462105263157895, 205.951578947368432, 213.069473684210521,
        220.187368421052611, 227.305263157894757, 232.507368421052632, 236.235789473684207,
        239.964210526315810, 243.692631578947385, 249.821052631578937, 259.069473684210550,
        268.317894736842106, 277.566315789473663, 290.926315789473733, 313.684210526315780,
        336.442105263157828, 359.200000000000045, 379.709473684210536, 389.538947368421020,
        399.368421052631618, 409.197894736842159, 418.048421052631625, 405.362105263157844,
        392.675789473684176, 379.989473684210566, 367.303157894736785, 362.862105263157900,
        359.206315789473706, 355.550526315789455, 351.894736842105317, 364.116842105263174,
        380.749473684210614, 397.382105263157825, 414.014736842105265, 425.200000000000045,
        433.479999999999961, 441.759999999999991, 450.040000000000020, 451.448421052631602,
        446.557894736842115, 441.667368421052629, 436.776842105263199, 427.680000000000007,
        412.039999999999964, 396.400000000000091, 380.759999999999991, 364.343157894736748,
        345.725263157894801, 327.107368421052627, 308.489473684210452, 289.202105263158046,
        265.451578947368432, 241.701052631578818, 217.950526315789602, 194.199999999999989]

PD_values = [float(value) for value in PD_data]

PD = {t: PD_values[t - 1] for t in Ωt}

fPV_data = [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
            0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
            0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
            0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
            0.000000000000000, 0.000010382119683, 0.000040230713771, 0.000070079307859, 0.000099927901947,
            0.000741167988464, 0.003583417447729, 0.006425666906994, 0.009267916366258, 0.013890410958904,
            0.037205479452055, 0.060520547945205, 0.083835616438356, 0.107150684931507, 0.147267599134823,
            0.188148233597693, 0.229028868060562, 0.269909502523432, 0.306336438356164, 0.341825753424658,
            0.377315068493151, 0.412804383561644, 0.440898399423216, 0.465756971881759, 0.490615544340303,
            0.515474116798846, 0.534759653929344, 0.549758240807498, 0.564756827685652, 0.579755414563807,
            0.590886373467916, 0.596989416005768, 0.603092458543619, 0.609195501081471, 0.611926027397260,
            0.606947945205479, 0.601969863013699, 0.596991780821918, 0.590157144917087, 0.574503878875270,
            0.558850612833453, 0.543197346791637, 0.526965306416727, 0.498000230713771, 0.469035155010815,
            0.440070079307859, 0.411105003604903, 0.375997087238645, 0.340304138428263, 0.304611189617880,
            0.268918240807499, 0.237489747656813, 0.207245825522711, 0.177001903388609, 0.146757981254506,
            0.122156308579668, 0.100563835616439, 0.078971362653208, 0.057378889689978, 0.042477000720981,
            0.033708147080029, 0.024939293439077, 0.016170439798125, 0.009808940158616, 0.007192213410238,
            0.004575486661860, 0.001958759913482, 0.000023100216294, 0.000017130497477, 0.000011160778659,
            0.000005191059841, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
            0.000000000000000]
fPV = {t: fPV_data[t - 1] for t in Ωt}

cOS_data = [0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.464400000000000, 0.499989473684211, 0.535578947368421, 0.571168421052631, 0.636063157894737, 0.713778947368421, 0.791494736842105, 0.869210526315790, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.903000000000000, 0.879347368421052, 0.801631578947368, 0.723915789473684, 0.646200000000000, 0.575810526315789, 0.540221052631579, 0.504631578947368, 0.469042105263158, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000, 0.435000000000000]
cOS_values = [float(value) for value in cOS_data]

# Create cOS dictionary
cOS = {t: cOS_values[t - 1] for t in Ωt}

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
    365 * gp.quicksum(p[c] * Δt * cOS[t] * PS[t, c, a] for t in Ωt for c in Ωc for a in Ωa) +
    365 * gp.quicksum(p[c] * Δt * cOT * PGD[t, c, a] for t in Ωt for c in Ωc for a in Ωa) +
    365 * gp.quicksum(p[c] * Δt * cCC * PD[t] * xD[t, c, a] for t in Ωt for c in Ωc for a in Ωa),
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
                PS[t, c, a] + PGD[t, c, a] + fPV[t] * PPVmax + PAEd[t, c, a] + PEVd[t, c, a] ==
                PD[t] * (1 - xD[t, c, a]) + PAEc[t, c, a] + PEVc[t, c, a],
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

# Assuming Ωc, Ωa, and Ωt are defined somewhere before this code

# Assuming Ωc, Ωa, and Ωt are defined somewhere before this code

num_c = len(Ωc)
num_a = len(Ωa)

# Create a 2D grid of subplots
fig, axes = plt.subplots(num_c, num_a, figsize=(15, 8), sharex=True)

# Flatten the 2D array of subplots for easy indexing
axes = axes.flatten()

# Loop over combinations of c and a
for i, c in enumerate(Ωc):
    for j, a in enumerate(Ωa):
        # Power-related plots
        axes[i * num_a + j].plot(Ωt, [PS_values[t, c, a] for t in Ωt], label="PS")
        axes[i * num_a + j].plot(Ωt, [PGD_values[t, c, a] for t in Ωt], label="PGD")
        axes[i * num_a + j].plot(Ωt, [PD[t] for t in Ωt], label="PD")
        axes[i * num_a + j].plot(Ωt, [fPV[t] * PPVmax_value for t in Ωt], label="fPV")
        axes[i * num_a + j].plot(Ωt, [PAEc_values[t, c, a] for t in Ωt], label="PAEc")
        axes[i * num_a + j].plot(Ωt, [-1 * PAEd_values[t, c, a] for t in Ωt], label="PAEd")
        axes[i * num_a + j].plot(Ωt, [PEVc_values[t, c, a] for t in Ωt], label="PEVc")
        axes[i * num_a + j].plot(Ωt, [-1 * PEVd_values[t, c, a] for t in Ωt], label="PEVd")
        axes[i * num_a + j].legend()
        axes[i * num_a + j].set_xlabel("Timestamp")
        axes[i * num_a + j].set_ylabel("Power [kW]")
        axes[i * num_a + j].set_title(f"Power Values - c{c}, a{a}")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()