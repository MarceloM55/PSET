from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import time
import json
import os
import matplotlib as mpl

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
πa = 0.1

start = time.time()

# model_name = "IonLitLFPInd"
model_name = "IonLitLFPIndNorm"

# Create a Gurobi model
model = gp.Model(par[model_name]['model name'])


# Variables
PPVmax = model.addVar(name="PPVmax", lb=0)
PGDmax = model.addVar(name="PGDmax", lb=0)
PAEmax = model.addVar(name="PAEmax", lb=0)
EAEmax = model.addVar(name="EAEmax", lb=0)

OPEX = model.addVar(name="OPEX", lb=0)
CAPEX = model.addVar(name="CAPEX", lb=0)
OPEX_yearly = model.addVar(name="OPEX", lb=0)


PS   = {(t, s, c, a): model.addVar(name=f"PS_{t}_{s}_{c}_{a}", lb=par[model_name]['PSmin'], ub=par[model_name]['PSmax'])    for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSp   = {(t, s, c, a): model.addVar(name=f"PSp_{t}_{s}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a
PSn   = {(t, s, c, a): model.addVar(name=f"PSn_{t}_{s}_{c}_{a}", lb=0)                  for t in Ωt for s in Ωs for c in Ωc for a in Ωa} # Substation Power in time t, contingency c, and EV scenario a


xD   = {(t, s, c, a): model.addVar(name=f"xD_{t}_{s}_{c}_{a}", lb=0, ub=1)              for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PGD  = {(t, s, c, a): model.addVar(name=f"PGD_{t}_{c}_{a}", lb=0, ub=par[model_name]['MaxGD'])     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

PAEc = {(t, s, c, a): model.addVar(name=f"PAEi_{t}_{s}_{c}_{a}", lb=0)                     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PAEd = {(t, s, c, a): model.addVar(name=f"PAEe_{t}_{s}_{c}_{a}", lb=0)                     for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
EAE  = {(t, s, c, a): model.addVar(name=f"EAE_{t}_{s}_{c}_{a}", lb=0)                      for t in Ωt for s in Ωs for c in Ωc for a in Ωa}



PEVc = {(t, s, c, a): model.addVar(name=f"PAEi_{t}_{s}_{c}_{a}", lb=0, ub=par[model_name]['EVPmaxc'])  for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
PEVd = {(t, s, c, a): model.addVar(name=f"PAEe_{t}_{s}_{c}_{a}", lb=0, ub=par[model_name]['EVPmaxd'])  for t in Ωt for s in Ωs for c in Ωc for a in Ωa}
SoCEV  = {(t, s, c, a): model.addVar(name=f"EEV_{t}_{s}_{c}_{a}", lb=0)                    for t in Ωt for s in Ωs for c in Ωc for a in Ωa}


 

# Objective function
model.setObjective(OPEX + CAPEX, GRB.MINIMIZE)

# Constraints
model.addConstr(OPEX_yearly == 
    365 * gp.quicksum(πs[s] * πc[c] * πa * Δt * par[model_name]['cOS'][t-1] * PSp[t, s, c, a]    for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * πa * Δt * (par[model_name]['cOT'] + par[model_name]['GDM']) * PGD[t, s, c, a]            for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * πa * Δt * ((par[model_name]['EAM'] * (PAEd[t, s, c, a] + PAEc[t, s, c, a])/2)  + (par[model_name]['EVM'] * (PEVd[t, s, c, a])/2))         for t in Ωt for s in Ωs for c in Ωc for a in Ωa) +
    365 * gp.quicksum(πs[s] * πc[c] * πa * Δt * par[model_name]['cCC'] * par[model_name]['MaxL'] * fp[s]["load"][t-1] * xD[t, s, c, a] for t in Ωt for s in Ωs for s in Ωs for c in Ωc for a in Ωa),
    name="OPEX_yearly"
)

model.addConstr(CAPEX == par[model_name]['cIPV'] * PPVmax + par[model_name]['cIT'] * PGDmax + par[model_name]['cIPA'] * PAEmax + par[model_name]['cIEA'] * EAEmax, name="CAPEX")
model.addConstr(OPEX == gp.quicksum( (1/(1+par[model_name]['rate'])**y) * OPEX_yearly for y in range(1,par[model_name]['nyears']+1)))

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
                        model.addConstr(PEVd[t, s, c, a] <= par[model_name]['EVPmaxd'] - (par[model_name]['EVPmaxd']/par[model_name]['EVPmaxc'])*PEVc[t, s, c, a], name=f"EV_Discharge_Max_{t}_{c}_{a}")
                    
                    elif t == Ωa[a]['arrival'][n]:
                        model.addConstr(SoCEV[t, s, c, a] == Ωa[a]['SoCini'][n], name=f"EV_SoC_ini_{t}_{c}_{a}")
                    elif n < len(Ωa[a]['arrival']) - 1:
                        if t > Ωa[a]['departure'][n] and t < Ωa[a]['arrival'][n+1]:
                            model.addConstr(SoCEV[t, s, c, a] == 0, name=f"EV_SoC_between_{t}_{c}_{a}")
                            model.addConstr(PEVc[t, s, c, a] == 0, name=f"EV_Charge_between_{t}_{c}_{a}")
                            model.addConstr(PEVd[t, s, c, a] == 0, name=f"EV_Discharge_between_{t}_{c}_{a}")

                # model.addConstr(PEVc[t,c,a] <= par[model_name]['EVPmaxc'], name=f"EV_ChargeMax_{t}_{c}_{a}")
                # model.addConstr(PEVd[t,c,a] <= par[model_name]['EVPmaxd'], name=f"EV_DischargeMax_{t}_{c}_{a}")
    


# Active power balance constraint
for t in Ωt:
    for s in Ωs:
        for c in Ωc:
            for a in Ωa:
                model.addConstr(
                    PS[t, s, c, a] + PGD[t, s, c, a] + fp[s]['pv'][t-1] * PPVmax + PAEd[t, s, c, a] + PEVd[t, s, c, a] ==
                    par[model_name]['MaxL']*fp[s]['load'][t-1] * (1 - xD[t, s, c, a]) + PAEc[t, s, c, a] + PEVc[t, s, c, a],
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
                        EAE[t, s, c, a] == EAE[t - 1, s, c, a] + par[model_name]['alpha'] * Δt * PAEc[t, s, c, a] - Δt * PAEd[t, s, c, a] / par[model_name]['alpha'] - EAE[t, s, c, a] * par[model_name]['beta'],
                        name=f"Energy_Storage_Balance_{t}_{s}_{c}_{a}"
                    )
                    model.addConstr(PAEc[t, s, c, a] <= (EAEmax - EAE[t-1, s, c, a])/(par[model_name]['alpha'] * Δt), name=f"Max_Charge_BESS_1_{t}_{s}_{c}_{a}")
                    model.addConstr(PAEd[t, s, c, a] <= EAE[t-1, s, c, a]*par[model_name]['alpha']/Δt, name=f"Max_Discharge_BESS_1_{t}_{s}_{c}_{a}")
                    
                if t == 1:
                    model.addConstr(
                        EAE[t, s, c, a] == par[model_name]['EAE0'] * EAEmax + par[model_name]['alpha'] * Δt * PAEc[t, s, c, a] - Δt * PAEd[t, s, c, a] / par[model_name]['alpha'],
                        name=f"Initial_Energy_Storage_initial_{t}_{s}_{c}_{a}"
                    )
                    model.addConstr(PAEc[t, s, c, a] <= (EAEmax - par[model_name]['EAE0'])/(par[model_name]['alpha'] * Δt), name=f"Max_Charge_BESS_2_{t}_{s}_{c}_{a}")
                    model.addConstr(PAEd[t, s, c, a] <= par[model_name]['EAE0']*par[model_name]['alpha']/Δt, name=f"Max_Discharge_BESS_2_{t}_{s}_{c}_{a}")

                model.addConstr(PAEd[t, s, c, a] <= PAEmax - PAEc[t, s, c, a], name=f"Max_Discharge_BESS_General_{t}_{s}_{c}_{a}")
                model.addConstr(EAE[t, s, c, a] <= EAEmax, name=f"Max_Energy_Storage_Capacity_{t}_{s}_{c}_{a}")
                model.addConstr(PAEc[t, s, c, a] <= PAEmax, name=f"Max_Injection_Power_{t}_{s}_{c}_{a}")
                model.addConstr(PAEd[t, s, c, a] <= PAEmax, name=f"Max_Extraction_Power_{t}_{s}_{c}_{a}")
                model.addConstr(PAEmax <= EAEmax * 0.5, name=f"Max_Injection_Power_{t}_{s}_{c}_{a}")


           

# Contingency operation constraint
for c in Ωc:
    for s in Ωs:
        for t in range(c, min(max(Ωt), c + int(par[model_name]['D'] / Δt)) + 1):
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

EAE_norm  = {(t, s, c, a): EAE[t, s, c, a].x/EAEmax_value for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

mpl.rc('font',family = 'serif', serif = 'cmr10')
plt.rcParams['axes.unicode_minus'] = False

EAE_norm  = {(t, s, c, a): EAE[t, s, c, a].x/EAEmax_value for t in Ωt for s in Ωs for c in Ωc for a in Ωa}

# Verify if folder exists
if not os.path.exists('Results'):
    os.makedirs('Results')

if not os.path.exists('Results/SoC'):
    os.makedirs('Results/SoC')

if not os.path.exists('Results/SoC/png'):
    os.makedirs('Results/SoC/png')

if not os.path.exists('Results/SoC/svg'):
    os.makedirs('Results/SoC/svg')

if not os.path.exists('Results/Operation'):
    os.makedirs('Results/Operation')

if not os.path.exists('Results/Operation/png'):
    os.makedirs('Results/Operation/png')

if not os.path.exists('Results/Operation/svg'):
    os.makedirs('Results/Operation/svg')

# Função para converter os dicionários em matrizes para visualização
def dict_to_matrix(values_dict, Ωt, s, c, Ωa):
    matrix = np.zeros((len(Ωa), len(Ωt)))
    for i, a in enumerate(Ωa):
        for j, t in enumerate(Ωt):
            key = (t, s, c, a)
            if key in values_dict:
                matrix[i, j] = values_dict[key]
    return matrix

# Gerar labels de horários de 00:00 até 23:55 de 15 em 15 minutos
times = [f'{hour:02d}:{minute:02d}' for hour in range(24) for minute in range(0, 60, 15)]

# for s in Ωs:
#     for c in Ωc:
#         # Converter dicionários em matrizes para a combinação atual de s e c
#         SoCEV_matrix = dict_to_matrix(SoCEV_values, Ωt, s, c, Ωa)
#         EAE_norm_matrix = dict_to_matrix(EAE_norm, Ωt, s, c, Ωa)

#         # Criar e configurar os gráficos
#         fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

#         # Configurar os subplots
#         for ax, matrix, title in zip(axs, [SoCEV_matrix, EAE_norm_matrix], ['EV SoC', 'BESS SoC']):
#             im = ax.imshow(matrix, aspect='auto', cmap='viridis')
#             ax.set_title(f'{title} for s={s}, c={c}')
#             ax.set_xlabel('Time')
#             ax.set_ylabel('EV scheduling scenarios')
#             ax.set_xticks(np.arange(0, len(times), 4))  # Colocar um tick a cada hora
#             ax.set_xticklabels(times[::4], rotation=45, ha='right')  # Rotacionar labels para melhor visualização
#             ax.set_yticks(np.arange(len(Ωa)))
#             ax.set_yticklabels(Ωa)

#         # Barra de cores mais fina e próxima
#         cbar = fig.colorbar(im, ax=axs, label='Value', shrink=0.8, pad=0.02, aspect=20)

#         # Salvar a figura com um nome único baseado em s e c
#         plt.savefig(f'Results/SoC/png/soc_s{s}_c{c}.png')
#         plt.savefig(f'Results/SoC/svg/soc_s{s}_c{c}.svg')
#         plt.close(fig)  # Fechar a figura para liberar memória

for s in Ωs:
    for c in Ωc:
        # Converter dicionários em matrizes para a combinação atual de s e c
        SoCEV_matrix = dict_to_matrix(SoCEV_values, Ωt, s, c, Ωa)
        EAE_norm_matrix = dict_to_matrix(EAE_norm, Ωt, s, c, Ωa)

        # Criar e configurar os gráficos
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
        # Ajuste dos dados para uso com pcolormesh
        x_edges = np.arange(SoCEV_matrix.shape[1])
        y_edges = np.arange(SoCEV_matrix.shape[0])

        # Plotar o mapa de calor para SoCEV_values usando pcolormesh
        ax = axs[0]
        matrix_pcolormesh = np.zeros((SoCEV_matrix.shape[0], SoCEV_matrix.shape[1]))
        matrix_pcolormesh[:-1, :-1] = SoCEV_matrix
        socev_im = ax.pcolormesh(x_edges, y_edges, matrix_pcolormesh, cmap='viridis', edgecolors='w', linewidth=0.01)
        ax.set_title(f'EV SoC for s={s}, c={times[c]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('EV scheduling scenarios')
        ax.set_xticks(np.arange(0, len(times), 4) + 0.5)
        ax.set_xticklabels(times[::4], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(Ωa)) + 0.5)
        ax.set_yticklabels(Ωa)

        # Plotar o mapa de calor para EAE_norm usando pcolormesh
        ax = axs[1]
        matrix_pcolormesh = np.zeros((EAE_norm_matrix.shape[0], EAE_norm_matrix.shape[1]))
        matrix_pcolormesh[:-1, :-1] = EAE_norm_matrix
        ax.pcolormesh(x_edges, y_edges, matrix_pcolormesh, cmap='viridis', edgecolors='w', linewidth=0.01)
        ax.set_title(f'BESS SoC for s={s}, c={times[c]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('EV scheduling scenarios')
        ax.set_xticks(np.arange(0, len(times), 4) + 0.5)
        ax.set_xticklabels(times[::4], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(Ωa)) + 0.5)
        ax.set_yticklabels(Ωa)

        # Criar a barra de cores com base no SoCEV_values
        cbar = fig.colorbar(socev_im, ax=axs, label='Value', shrink=0.8, pad=0.02, aspect=20)

        # Salvar a figura com um nome único baseado em s e c
        plt.savefig(f'Results/SoC/png/soc_s{s}_c{c}.png')
        plt.savefig(f'Results/SoC/svg/soc_s{s}_c{c}.svg')
        plt.close(fig)  # Fechar a figura para liberar memória


for s in Ωs:
    for c in Ωc:
        for a in Ωa:
            plt.figure(figsize=(8, 4))
            plt.plot(times, [PS_values[t, s, c, a] for t in Ωt], label="EDS", color='blue')
            plt.plot(times, [PGD_values[t, s, c, a] for t in Ωt], label="TG", color='red')
            plt.plot(times, [fp[s]['load'][t-1] * par['IonLitResNCAInd']['MaxL'] * (1 - xD_values[t,s,c,a]) for t in Ωt], label="Demand", color='black', marker='o', linestyle='dashed')
            plt.plot(times, [-1*fp[s]['pv'][t-1] * PPVmax_value  for t in Ωt], label="PV", color='orange')
            if EAEmax_value > 0:
                plt.bar(Ωt, [PAEc_values[t, s, c, a] - PAEd_values[t, s, c, a] for t in Ωt], label="BESS", color='green')
            plt.plot(Ωt, [PEVc_values[t, s, c, a] - PEVd_values[t, s, c, a] for t in Ωt], label="EV", color='purple')
            plt.legend()
            plt.xticks(np.arange(0, len(times), 4), times[::4], rotation=45)
            plt.xticks(fontsize=10)  # Set font size for x-axis ticks
            plt.yticks(fontsize=10) 
            plt.xlabel("Timestamp")
            plt.ylabel("Power [kW]")
            plt.tight_layout()
            plt.savefig(f"Results/Operation/svg/operation_s{s}_c{c}_a{a}.svg")
            plt.savefig(f"Results/Operation/png/operation_s{s}_c{c}_a{a}.png")
            plt.close()