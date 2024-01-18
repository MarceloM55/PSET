import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np



# Now you can use x_values as a NumPy array


delta = 0.25  # Define the time interval in hours

# Create a Gurobi model
model = gp.Model("Microgrid Optimization")

# Sets and Parameters
T = range(1, 97)  # Set of time intervals
C = [0, 72, 80, 88]  # Set of operation scenarios
A = {'TillMidday': list(T[:48]), 'EightToSix': list(T[:32]) + list(T[72:]), 'OnlyAtNight': list(T[80:])} # EV scenarios

# Assuming T is the list of time intervals, you can use the scenarios like this:

# Scenario: TillMidday - EV available till midday
scenario_hours_till_midday = A['TillMidday']

# Scenario: EightToSix - EV available from 8 AM to 6 PM
scenario_hours_eight_to_six = A['EightToSix']

# Scenario: OnlyAtNight - EV available only at night
scenario_hours_only_at_night = A['OnlyAtNight']

#PD_data = [158.9, 157.7, 180.0, 180.3, 184.7, 159.1, 130.3, 98.5, 105.9, 193.7, 201.0, 230.4, 245.8,
#      284.0, 378.0, 418.6, 366.2, 351.1, 419.8, 454.0, 433.8, 369.2, 292.3, 194.2]
PD_data = ['158.900000000000006', '158.609473684210542', '158.318947368421050', '158.028421052631586',
        '157.737894736842094', '162.394736842105260', '167.793684210526322', '173.192631578947356',
        '178.591578947368419', '180.053684210526313', '180.126315789473693', '180.198947368421074',
        '180.271578947368425', '180.948421052631574', '182.013684210526321', '183.078947368421041',
        '184.144210526315788', '181.735789473684207', '175.537894736842077', '169.340000000000003',
        '163.142105263157902', '156.674736842105261', '149.702105263157904', '142.729473684210518',
        '135.756842105263161', '128.626315789473693', '120.927368421052634', '113.228421052631575',
        '105.529473684210529', '98.655789473684209', '100.447368421052630', '102.238947368421051',
        '104.030526315789473', '105.822105263157908', '126.232631578947348', '147.489473684210452',
        '168.746315789473726', '190.003157894736830', '195.159999999999997', '196.927368421052620',
        '198.694736842105272', '200.462105263157895', '205.951578947368432', '213.069473684210521',
        '220.187368421052611', '227.305263157894757', '232.507368421052632', '236.235789473684207',
        '239.964210526315810', '243.692631578947385', '249.821052631578937', '259.069473684210550',
        '268.317894736842106', '277.566315789473663', '290.926315789473733', '313.684210526315780',
        '336.442105263157828', '359.200000000000045', '379.709473684210536', '389.538947368421020',
        '399.368421052631618', '409.197894736842159', '418.048421052631625', '405.362105263157844',
        '392.675789473684176', '379.989473684210566', '367.303157894736785', '362.862105263157900',
        '359.206315789473706', '355.550526315789455', '351.894736842105317', '364.116842105263174',
        '380.749473684210614', '397.382105263157825', '414.014736842105265', '425.200000000000045',
        '433.479999999999961', '441.759999999999991', '450.040000000000020', '451.448421052631602',
        '446.557894736842115', '441.667368421052629', '436.776842105263199', '427.680000000000007',
        '412.039999999999964', '396.400000000000091', '380.759999999999991', '364.343157894736748',
        '345.725263157894801', '327.107368421052627', '308.489473684210452', '289.202105263158046',
        '265.451578947368432', '241.701052631578818', '217.950526315789602', '194.199999999999989']
PD_values = [float(value) for value in PD_data]

PD = {t: PD_values[t - 1] for t in T}
#fPV_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.000403478, 0.021839784, 0.162691836, 0.371420561, 0.561875517, # dados originais
#            0.667180143, 0.728490720, 0.715147406, 0.673689545, 0.585161204, 0.441333950, 0.274392455,
#            0.132082341, 0.088886594, 0.010857717, 0.000000000, 0.000000000, 0.000000000, 0.000000000]
#fPV_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001232876712328768, 0.011863013698630123, 0.10816438356164386, #dados de 2015 de Porto
#            0.2770191780821916, 0.42360547945205484, 0.5262821917808218, 0.5882328767123286, 0.6134410958904107,
#            0.5928794520547946, 0.5282246575342465, 0.4085863013698635, 0.2611589041095892, 0.13623835616438362,
#            0.04705205479452055, 0.010832876712328762, 2.4657534246575345e-05, 0.0, 0.0]
fPV_data = ['0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000',
            '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000',
            '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000',
            '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000',
            '0.000000000000000', '0.000010382119683', '0.000040230713771', '0.000070079307859', '0.000099927901947',
            '0.000741167988464', '0.003583417447729', '0.006425666906994', '0.009267916366258', '0.013890410958904',
            '0.037205479452055', '0.060520547945205', '0.083835616438356', '0.107150684931507', '0.147267599134823',
            '0.188148233597693', '0.229028868060562', '0.269909502523432', '0.306336438356164', '0.341825753424658',
            '0.377315068493151', '0.412804383561644', '0.440898399423216', '0.465756971881759', '0.490615544340303',
            '0.515474116798846', '0.534759653929344', '0.549758240807498', '0.564756827685652', '0.579755414563807',
            '0.590886373467916', '0.596989416005768', '0.603092458543619', '0.609195501081471', '0.611926027397260',
            '0.606947945205479', '0.601969863013699', '0.596991780821918', '0.590157144917087', '0.574503878875270',
            '0.558850612833453', '0.543197346791637', '0.526965306416727', '0.498000230713771', '0.469035155010815',
            '0.440070079307859', '0.411105003604903', '0.375997087238645', '0.340304138428263', '0.304611189617880',
            '0.268918240807499', '0.237489747656813', '0.207245825522711', '0.177001903388609', '0.146757981254506',
            '0.122156308579668', '0.100563835616439', '0.078971362653208', '0.057378889689978', '0.042477000720981',
            '0.033708147080029', '0.024939293439077', '0.016170439798125', '0.009808940158616', '0.007192213410238',
            '0.004575486661860', '0.001958759913482', '0.000023100216294', '0.000017130497477', '0.000011160778659',
            '0.000005191059841', '0.000000000000000', '0.000000000000000', '0.000000000000000', '0.000000000000000',
                              '0.000000000000000']
fPV = {t: fPV_data[t - 1] for t in T}
#cOS_data = [0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435, 0.435,
#            0.582, 0.903, 0.903, 0.903, 0.582, 0.435, 0.435, 0.435, 0.435, 0.435,
#            0.435, 0.435, 0.435, 0.435]
cOS_data = ['0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.464400000000000', '0.499989473684211', '0.535578947368421', '0.571168421052631', '0.636063157894737', '0.713778947368421', '0.791494736842105', '0.869210526315790', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.903000000000000', '0.879347368421052', '0.801631578947368', '0.723915789473684', '0.646200000000000', '0.575810526315789', '0.540221052631579', '0.504631578947368', '0.469042105263158', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000', '0.435000000000000']
cOS_values = [float(value) for value in cOS_data]

# Create cOS dictionary
cOS = {t: cOS_values[t - 1] for t in T}

p = {0: 0.985, 72: 0.005, 80: 0.005, 88: 0.005}

pEV = {'TillMidday': 0.25, 'EightToSix': 0.5, 'OnlyAtNight': 0.25}

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
EEV0 = 0

#EV maximum charge
MaxEVCharge = 700

# Define maximum charging and discharging power EV
MaxChargePower = 500  # Adjust this value based on your requirements
MaxDischargePower = 500  # Adjust this value based on your requirements

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0, ub=IPPVmax)
PGDmax = model.addVar(name="PGDmax", lb=0, ub=IPGDmax)
PAEmax = model.addVar(name="PAEmax", lb=0) #Both this and bottom one could be selected from a pool
EAEmax = model.addVar(name="EAEmax", lb=0)
PS = {(t, c, a): model.addVar(name=f"PS_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A} # Substation Power in time t, contingency c, and EV scenario a
PGD = {(t, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}
xD = {(t, c, a): model.addVar(name=f"xD_{t}_{c}_{a}", lb=0, ub=1) for t in T for c in C for a in A}
PAEi = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}
PAEe = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}
EAE = {(t, c, a): model.addVar(name=f"EAE_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}

PEVi = {(t, c, a): model.addVar(name=f"PAEi_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}
PEVe = {(t, c, a): model.addVar(name=f"PAEe_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}
EEV = {(t, c, a): model.addVar(name=f"EEV_{t}_{c}_{a}", lb=0) for t in T for c in C for a in A}

# Define binary decision variables for EV availability scenarios
CEV = {a: model.addVars(A[a], vtype=GRB.BINARY, name=f"BatteryAvailability_{a}") for a in A}

# Define binary decision variables for EV availability scenarios
EVei = {(t, c, a): model.addVar(vtype=GRB.BINARY, name=f"EVBatteryChargeDischarge{a}") for t in T for c in C for a in A} # For EVei = 1, only PEVe > 0, and for EVei = 0, only PVei

# Objective function
model.setObjective(
    cIPV * PPVmax + cIT * PGDmax + cIPA * PAEmax + cIEA * EAEmax +
    365 * gp.quicksum(p[c] * delta * cOS[t] * PS[t, c, a] for t in T for c in C for a in A) +
    365 * gp.quicksum(p[c] * delta * cOT * PGD[t, c, a] for t in T for c in C for a in A) +
    365 * gp.quicksum(p[c] * delta * cCC * PD[t] * xD[t, c, a] for t in T for c in C for a in A),
    GRB.MINIMIZE
)

# Assuming T is the list of time intervals
for t in T:
    for c in C:
        for a in A:
            if t not in scenario_hours_till_midday:
                model.addConstr(PEVe[t, c, 'TillMidday'] == 0, f"NoCharging_TillMidday_{t}")
                model.addConstr(PEVi[t, c, 'TillMidday'] == 0, f"NoDischarging_TillMidday_{t}")

            if t not in scenario_hours_eight_to_six:
                model.addConstr(PEVe[t, c, 'EightToSix'] == 0, f"NoCharging_EightToSix_{t}")
                model.addConstr(PEVi[t, c, 'EightToSix'] == 0, f"NoDischarging_EightToSix_{t}")

            if t not in scenario_hours_only_at_night:
                model.addConstr(PEVe[t, c, 'OnlyAtNight'] == 0, f"NoCharging_OnlyAtNight_{t}")
                model.addConstr(PEVi[t, c, 'OnlyAtNight'] == 0, f"NoDischarging_OnlyAtNight_{t}")

# Assuming you have a variable TempBatteryAvailable indicating the availability of the temporary battery
for t in T:
    for c in C:
        for a in A:
            if t == 48 and a == 'TillMidday':
                model.addConstr(EEV[t, c, 'TillMidday'] >= 0.75 * MaxEVCharge, f"minimumExitCharge{t}")
            if t == 32 and a == 'EightToSix':
                model.addConstr(EEV[t, c, 'EightToSix'] >= 0.75 * MaxEVCharge, f"minimumExitCharge{t}")

# Constraints to ensure the EV is used only during specific hours for each scenario
for a in A:
    for c in C:
        for t in CEV[a]:
            model.addConstr(xD[t, c, a] <= CEV[a][t])

# Add constraints for maximum charging and discharging power
for t in T:
    for c in C:
        for a in A:
            model.addConstr(PEVi[t, c,  a] <= MaxChargePower)  # Maximum charging power
            model.addConstr(PEVe[t, c, a] <= MaxDischargePower)  # Maximum discharging power

# Active power balance constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                PS[t, c, a] + PGD[t, c, a] + fPV[t] * PPVmax ==
                PD[t] * (1 - xD[t, c, a]) + PAEe[t, c, a] - PAEi[t, c, a] + PEVe[t, c, a] - PEVi[t, c, a],
                name=f"Active_Power_Balance_{t}_{c}_{a}"
        )



# Substation capacity constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                PS[t, c, a] <= PSmax,
                name=f"Substation_Capacity_{t}_{c}_{a}"
        )

# Conventional generator capacity constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                PGD[t, c, a] <= PGDmax,
                name=f"Generator_Capacity_{t}_{c}_{a}"
        )

# Max injection power from storage constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                PAEi[t, c, a] <= PAEmax,
                name=f"Max_Injection_Power_{t}_{c}+{a}"
        )

# Max extraction power from storage constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                PAEe[t, c, a] <= PAEmax,
                name=f"Max_Extraction_Power_{t}_{c}_{a}"
        )

# Energy storage balance constraint
for t in T:
    for c in C:
        for a in A:
            if t > 1:
                model.addConstr(
                    EAE[t, c, a] == EAE[t - 1, c, a] + alpha * delta * PAEe[t, c, a] - delta * PAEi[t, c, a] / alpha - beta * delta * EAE[t, c, a],
                    name=f"Energy_Storage_Balance_{t}_{c}_{a}"
            )

# Initial energy storage constraint
for t in T:
    for c in C:
        for a in A:
            if t == 1:
                model.addConstr(
                    EAE[t, c, a] == EAE0 + alpha * delta * PAEe[t, c, a] - delta * PAEi[t, c, a] / alpha - beta * delta * EAE[t, c, a],
                    name=f"Initial_Energy_Storage_{t}_{c}"
                )



# Initial energy storage constraint
for t in T:
    for c in C:
        for a in A:
            if t == 1:
                model.addConstr(
                    EEV[t, c, a] == EEV0 + alpha * delta * PEVe[t, c, a] * EVei[t, c, a] - delta * PEVi[t, c, a] * (1 - EVei[t, c, a]) / alpha - beta * delta * EEV[t, c, a],
                    name=f"Initial_Energy_Storage_{t}_{c}"
                )

# Constraints to relate charging, discharging, and SOC
for t in T[1:]:
    for c in C:
        for a in A:
            model.addConstr(EEV[t, c, a] <= MaxEVCharge)  # Limit SOC within the battery capacity
            model.addConstr(PEVe[t, c, a] <= MaxChargePower) 
            model.addConstr(PEVi[t, c, a] <= MaxDischargePower) 
            if t > 1:
                model.addConstr(EEV[t, c, a] == EEV[t - 1, c, a] + alpha * delta * PEVe[t, c, a] * EVei[t, c, a] - delta * PEVi[t, c, a] * (1 - EVei[t, c, a]) / alpha - beta * delta * EEV[t, c, a])
            


# Max energy storage capacity constraint
for t in T:
    for c in C:
        for a in A:
            model.addConstr(
                EAE[t, c, a] <= EAEmax,
                name=f"Max_Energy_Storage_Capacity_{t}_{c}_{a}"
        )

# Contingency operation constraint
for c in C:
    for t in range(c, min(max(T), c + int(D / delta)) + 1):
        for a in A:
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

print(EEV)

# Extract the values for plotting
PS_values = {(t, c, a): PS[t, c, a].x for t in T for c in C for a in A}
PGD_values = {(t, c, a): PGD[t, c, a].x for t in T for c in C for a in A}

PEVi_values = {(t, c, a): PEVi[t, c, a].x for t in T for c in C for a in A}
PEVe_values = {(t, c, a): PEVe[t, c, a].x for t in T for c in C for a in A}

PAEi_values = {(t, c, a): PAEi[t, c, a].x for t in T for c in C for a in A}
PAEe_values = {(t, c, a): PAEe[t, c, a].x for t in T for c in C for a in A}
EAE_values = {(t, c, a): EAE[t, c, a].x for t in T for c in C for a in A}

# Assuming EEV is a Gurobi variable
EEV_values = {(t, c, a): EEV[t, c, a].x for t in T for c in C for a in A}

fig = plt.figure(figsize=(15, 10))

# Substation Power
ax1 = fig.add_subplot(331, projection='3d')
for a in A:
    for c in C:
        ax1.plot(T, [c] * len(T), [PS_values[t, c, a] for t in T], label=f"PS_{c}_{a}")

ax1.set_xlabel("Time (Interval)")
ax1.set_ylabel("Contingency")
ax1.set_zlabel("Substation Power")
ax1.set_title("Substation Power")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Generator Power
ax2 = fig.add_subplot(332, projection='3d')
for a in A:
    for c in C:
        ax2.plot(T, [c] * len(T), [PGD_values[t, c, a] for t in T], label=f"PGD_{c}_{a}")

ax2.set_xlabel("Time (Interval)")
ax2.set_ylabel("Contingency")
ax2.set_zlabel("Generator Power")
ax2.set_title("Generator Power")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# EV Charging/Discharging
ax3 = fig.add_subplot(333, projection='3d')
for a in A:
    for c in C:
        ax3.plot(T, [c] * len(T), [PEVi_values[t, c, a] for t in T], label=f"PEVi_{c}_{a}")

ax3.set_xlabel("Time (Interval)")
ax3.set_ylabel("Contingency")
ax3.set_zlabel("EV Charging")
ax3.set_title("EV Charging")
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Energy Storage
ax4 = fig.add_subplot(334, projection='3d')
for a in A:
    for c in C:
        ax4.plot(T, [c] * len(T), [EAE_values[t, c, a] for t in T], label=f"EAE_{c}_{a}")

ax4.set_xlabel("Time (Interval)")
ax4.set_ylabel("Contingency")
ax4.set_zlabel("Energy Storage")
ax4.set_title("Energy Storage")
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# EEV
ax5 = fig.add_subplot(335, projection='3d')
for a in A:
    for c in C:
        ax5.plot(T, np.full_like(T, c), EEV_values[t, c, a], label=f"EEV_{c}_{a}")

ax5.set_xlabel('Time Intervals')
ax5.set_ylabel('Contingency')
ax5.set_zlabel('EEV')
ax5.set_title("EEV Storage")
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# EV Discharging
ax6 = fig.add_subplot(336, projection='3d')
for a in A:
    for c in C:
        ax6.plot(T, [c] * len(T), [PEVe_values[t, c, a] for t in T], label=f"xE_{c}_{a}")

ax6.set_xlabel("Time (Interval)")
ax6.set_ylabel("Contingency")
ax6.set_zlabel("EV Discharging")
ax6.set_title("EV Discharging")
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# EV Charging/Discharging
ax7 = fig.add_subplot(337, projection='3d')
for a in A:
    for c in C:
        ax7.plot(T, [c] * len(T), [PAEe_values[t, c, a] for t in T], label=f"PAEe_{c}_{a}")

ax7.set_xlabel("Time (Interval)")
ax7.set_ylabel("Contingency")
ax7.set_zlabel("AE Charging")
ax7.set_title("AE Charging")
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# EV Charging/Discharging
ax8 = fig.add_subplot(338, projection='3d')
for a in A:
    for c in C:
        ax8.plot(T, [c] * len(T), [PAEi_values[t, c, a] for t in T], label=f"PAEi_{c}_{a}")

ax8.set_xlabel("Time (Interval)")
ax8.set_ylabel("Contingency")
ax8.set_zlabel("AE Discharging")
ax8.set_title("AE Discharging")
ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


model.dispose()
