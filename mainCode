import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

delta = 0.25  # Define the time interval in hours

# Create a Gurobi model
model = gp.Model("Microgrid Optimization")

# Sets and Parameters
T = range(1, 97)  # Set of time intervals
C = [0, 72, 80, 88]  # Set of operation scenarios

# Define EV scenarios
scenarios = {'TillMidday': T[:48], 'EightToSix': list(T[:32]) + list(T[72:]), 'OnlyAtNight': T[80:]}

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

#EV maximum charge
MaxEVCharge = 20

# Define maximum charging and discharging power EV
MaxChargePower = 10  # Adjust this value based on your requirements
MaxDischargePower = 15  # Adjust this value based on your requirements

# Variables
PPVmax = model.addVar(name="PPVmax", lb=0, ub=IPPVmax)
PGDmax = model.addVar(name="PGDmax", lb=0, ub=IPGDmax)
PAEmax = model.addVar(name="PAEmax", lb=0) #Both this and bottom one could be selected from a pool
EAEmax = model.addVar(name="EAEmax", lb=0)
PS = {(t, c): model.addVar(name=f"PS_{t}_{c}", lb=0) for t in T for c in C}
PGD = {(t, c): model.addVar(name=f"PGD_{t}_{c}", lb=0) for t in T for c in C}
xD = {(t, c): model.addVar(name=f"xD_{t}_{c}", lb=0, ub=1) for t in T for c in C}
PAEi = {(t, c): model.addVar(name=f"PAEi_{t}_{c}", lb=0) for t in T for c in C}
PAEe = {(t, c): model.addVar(name=f"PAEe_{t}_{c}", lb=0) for t in T for c in C}
EAE = {(t, c): model.addVar(name=f"EAE_{t}_{c}", lb=0) for t in T for c in C}

# Define the xD_Temp variable as a dictionary
xD_Temp = {}
for t in T:
    for s in scenarios:
        xD_Temp[t, s] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"xD_Temp_{t}_{s}")

# Define binary decision variable for the EV
TempBatteryAvailable = model.addVars(T, vtype=GRB.BINARY, name="TempBatteryAvailability")

# Define the SOC_Temp variable as a dictionary
SOC_Temp = {}
for t in T:
    for s in scenarios:
        SOC_Temp[t, s] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"SOC_Temp_{t}_{s}")

# Define binary decision variables for EV availability scenarios
CEV = {s: model.addVars(scenarios[s], vtype=GRB.BINARY, name=f"BatteryAvailability_{s}") for s in scenarios}

# Objective function
model.setObjective(
    cIPV * PPVmax + cIT * PGDmax + cIPA * PAEmax + cIEA * EAEmax +
    365 * gp.quicksum(p[c] * delta * cOS[t] * PS[t, c] for t in T for c in C) +
    365 * gp.quicksum(p[c] * delta * cOT * PGD[t, c] for t in T for c in C) +
    365 * gp.quicksum(p[c] * delta * cCC * PD[t] * xD[t, c] for t in T for c in C),
    GRB.MINIMIZE
)

# Constraints to ensure the EV is used only during specific hours for each scenario
for s in scenarios:
    for c in C:
        for t in CEV[s]:
            model.addConstr(xD[t, c] <= CEV[s][t])

# Add constraints for maximum charging and discharging power
for t in T:
    for s in scenarios:
        model.addConstr(xD_Temp[t, s] <= MaxChargePower)  # Maximum charging power
        model.addConstr(xD_Temp[t, s] >= -MaxDischargePower)  # Maximum discharging power

# Active power balance constraint
for t in T:
    for c in C:
        model.addConstr(
            PS[t, c] + PGD[t, c] + fPV[t] * PPVmax ==
            PD[t] * (1 - xD[t, c]) + PAEe[t, c] - PAEi[t, c],
            name=f"Active_Power_Balance_{t}_{c}"
        )

# Substation capacity constraint
for t in T:
    for c in C:
        model.addConstr(
            PS[t, c] <= PSmax,
            name=f"Substation_Capacity_{t}_{c}"
        )

# Conventional generator capacity constraint
for t in T:
    for c in C:
        model.addConstr(
            PGD[t, c] <= PGDmax,
            name=f"Generator_Capacity_{t}_{c}"
        )

# Max injection power from storage constraint
for t in T:
    for c in C:
        model.addConstr(
            PAEi[t, c] <= PAEmax,
            name=f"Max_Injection_Power_{t}_{c}"
        )

# Max extraction power from storage constraint
for t in T:
    for c in C:
        model.addConstr(
            PAEe[t, c] <= PAEmax,
            name=f"Max_Extraction_Power_{t}_{c}"
        )

# Energy storage balance constraint
for t in T:
    for c in C:
        if t > 1:
            model.addConstr(
                EAE[t, c] == EAE[t - 1, c] + alpha * delta * PAEe[t, c] - delta * PAEi[t, c] / alpha - beta * delta * EAE[t, c],
                name=f"Energy_Storage_Balance_{t}_{c}"
            )

# Initial energy storage constraint
for t in T:
    for c in C:
        if t == 1:
            model.addConstr(
                EAE[t, c] == EAE0 + alpha * delta * PAEe[t, c] - delta * PAEi[t, c] / alpha - beta * delta * EAE[t, c],
                name=f"Initial_Energy_Storage_{t}_{c}"
            )

# Initialize SOC at the first time step for each scenario
for s in scenarios:
    model.addConstr(SOC_Temp[T[0], s] == 0)

# Define the SOC dynamics
for t in T[1:]:
    for s in scenarios:
        if t in scenarios[s]:
            model.addConstr(SOC_Temp[t, s] == SOC_Temp[t - 1, s] + xD_Temp[t, s] * delta)

# Now you can use xD_Temp in your constraints
for t in T:
    for s in scenarios:
        model.addConstr(xD_Temp[t, s] <= TempBatteryAvailable[t])

# Assuming you have a variable TempBatteryAvailable indicating the availability of the temporary battery
for s in scenarios:
    model.addConstr(SOC_Temp[T[-1], s] >= 0.75 * MaxEVCharge)



# Constraints to relate charging, discharging, and SOC
for t in T[1:]:
    for s in scenarios:
        model.addConstr(SOC_Temp[t, s] == SOC_Temp[t - 1, s] + xD_Temp[t, s] * delta)
        model.addConstr(SOC_Temp[t, s] <= MaxEVCharge)  # Limit SOC within the battery capacity


# Max energy storage capacity constraint
for t in T:
    for c in C:
        model.addConstr(
            EAE[t, c] <= EAEmax,
            name=f"Max_Energy_Storage_Capacity_{t}_{c}"
        )

# Contingency operation constraint
for c in C:
    for t in range(c, min(max(T), c + int(D / delta)) + 1):
        if c != 0:
            model.addConstr(
                PS[t, c] == 0,
                name=f"Contingency_Operation_{c}_{t}"
            )

# Solve the model
model.optimize()




print(PPVmax)
print(PGDmax)
print(PAEmax)
print(EAEmax)

# Extract the values for plotting
PS_values = {(t, c): PS[t, c].x for t in T for c in C}
PGD_values = {(t, c): PGD[t, c].x for t in T for c in C}
xE_values = {(t, c): xD[t, c].x for t in T for c in C}
PAEi_values = {(t, c): PAEi[t, c].x for t in T for c in C}
PAEe_values = {(t, c): PAEe[t, c].x for t in T for c in C}
EAE_values = {(t, c): EAE[t, c].x for t in T for c in C}
xD_Temp_values = {(t, s): xD_Temp[t, s].x if (t, s) in xD_Temp else 0 for t in T for s in scenarios}

# Plotting
plt.figure(figsize=(15, 30))

# Plot PS for each scenario
plt.subplot(4, 2, 1)
for c in C:
    plt.plot(T, [PS_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('PS (Active Power from Substation)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend()

# Plot PGD for each scenario
plt.subplot(4, 2, 2)
for c in C:
    plt.plot(T, [PGD_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('PGD (Active Power from Conventional Generator)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend()

# Plot xE for each scenario
plt.subplot(4, 2, 3)
for c in C:
    plt.plot(T, [xE_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('xE (Percentage of Load Shedding)')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.legend()

# Plot PAEi for each scenario
plt.subplot(4, 2, 4)
for c in C:
    plt.plot(T, [PAEi_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('PAEi (Injected Power to Energy Storage)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend()

# Plot PAEe for each scenario
plt.subplot(4, 2, 5)
for c in C:
    plt.plot(T, [PAEe_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('PAEe (Extracted Power from Energy Storage)')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend()

# Plot EAE for each scenario
plt.subplot(4, 2, 6)
for c in C:
    plt.plot(T, [EAE_values[t, c] for t in T], label=f'Scenario {c}')
plt.title('EAE (Energy Stored in Energy Storage)')
plt.xlabel('Time')
plt.ylabel('Energy (kWh)')
plt.legend()

# Plot xD_Temp for each scenario
plt.subplot(4, 2, 7)
for s in scenarios:
    plt.plot(T, [xD_Temp_values[t, s] for t in T], label=f'Scenario {s}')
plt.title('xD_Temp (Temporary Battery Charge)')
plt.xlabel('Time')
plt.ylabel('Charge (kW)')
plt.legend()

plt.tight_layout()
plt.show()
# Dispose of the model
model.dispose()
