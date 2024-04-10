import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Assuming you have your data and parameters set up appropriately

# Loop over each scenario
for s in range(1, 11):  # Assuming you have 10 scenarios
    plt.figure()
    plt.plot(Ωt, [pyo.value(model.Peds[t, s]) for t in range(len(Ωt))], label='EDS', color='blue')
    plt.plot(Ωt, [-data['PV']['Pmax'] * fs[s]['pv'][t] for t in range(len(Ωt))], label='PV', color='orange')
    plt.plot(Ωt, [data['load']['Pmax'] * fs[s]['load'][t] for t in range(len(Ωt))], label='Load', color='black', marker='o', linestyle='dashed', markersize=3)

    for evcs, connector in [(evcs, connector) for evcs in range(1, 4) for connector in range(1, 4)]:  # Assuming you have 3 EVCSs with 3 connectors each
        vehicle_connected = [pyo.value(model.αEV[t, s, e, evcs, connector]) for t in range(len(Ωt)) for e in range(1, 11)]  # Assuming you have 10 EVs
        plt.plot(Ωt, vehicle_connected, label=f'{evcs}{connector}', linestyle='dashed')
        plt.fill_between(Ωt, 0, vehicle_connected, where=[v == 1 for v in vehicle_connected], color='red', alpha=0.3)

    plt.ylabel('Power [kW]')
    plt.xlabel('Timestamp')
    plt.legend(loc='upper right')
    plt.savefig(f'Results/Operation_{s}.png', dpi=300, bbox_inches='tight')
    plt.close('All')

    plt.figure()
    plt.plot(Ωt, [pyo.value(model.SoCbess[t, s]) for t in range(len(Ωt))], label='SoC BESS', color='blue')

    for e in range(1, 11):  # Assuming you have 10 EVs
        plt.plot(Ωt, [pyo.value(model.SoCEV[t, s, e]) for t in range(len(Ωt))], label=f'SoC EV {e}', linestyle='dashed', marker='o', markersize=1)

    plt.ylabel('State of Charge')
    plt.xlabel('Timestamp')
    plt.legend(loc='upper right')
    plt.savefig(f'Results/SoC_{s}.png', dpi=300, bbox_inches='tight')
    plt.close('All')

    # Heatmap for EVCSs
    dic_values = {}
    dic_annotations = {}

    for evcs, connector in [(evcs, connector) for evcs in range(1, 4) for connector in range(1, 4)]:  # Assuming you have 3 EVCSs with 3 connectors each
        values = []
        annotations = []

        for t in range(len(Ωt)):
            idev = 0
            idsoc = 0
            annotation = '0\n0'

            for e in range(1, 11):  # Assuming you have 10 EVs
                if pyo.value(model.αEV[t, s, e, evcs, connector]) == 1:
                    if idev > 0:
                        print(f'Error: More than one EV connected to {evcs}{connector} at {t}')
                    idev = e
                    idsoc = int(100 * pyo.value(model.SoCEV[t, s, e]))
                    annotation = f'{idev}\n{idsoc}%'
            if idev == 0:
                annotation = f'\n'

            values.append(idev)
            annotations.append(annotation)

        dic_values[f'{evcs}{connector}'] = values
        dic_annotations[f'{evcs}{connector}'] = annotations

    df_values = pd.DataFrame.from_dict(dic_values, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)
    df_annotations = pd.DataFrame.from_dict(dic_annotations, orient='index', columns=pd.to_datetime(Ωt, format='%H:%M').time)

    plt.figure(figsize=(16, 4))
    ax = sns.heatmap(df_values, cmap='tab20', cbar=False, linewidths=.5)

    for y, row in enumerate(df_annotations.values):
        for x, cell in enumerate(row):
            idev, idsoc = cell.split('\n')
            ax.text(x + 0.5, y + 0.3, idev, ha='center', va='center', fontsize=6)
            ax.text(x + 0.5, y + 0.7, idsoc, ha='center', va='center', fontsize=4, color='black', rotation=-90)

    plt.title(f'EVCSs - Scenario {s}')
    plt.xlabel('Timestamp')
    plt.ylabel('EVCSs')
    plt.savefig(f'Results/EVCS-s{s}.png', dpi=300, bbox_inches='tight')
    plt.close('All')
