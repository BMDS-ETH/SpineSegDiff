from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def segmentation_pathology_analysis(dice_df: pd.DataFrame, grading_df:  pd.DataFrame, save_path=None, title=None):
    # Merge data on patient ID
    scores_pathologies = pd.merge(dice_df, grading_df, on='pid')
    # Define spine structures and pathologies
    spine_structures = ['SC', 'VB', 'IVD']
    pathologies = grading_df.columns[1:-1]  # Exclude 'pid' and 'Pfirrman grade'
    fig, axes = plt.subplots(2, len(pathologies) // 2 + len(pathologies) % 2, figsize=(16, 9.6))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    statistical_test = {}

    # Combine "Low" and "Up" Endplate Pathologies
    #scores_pathologies['Endplate Changes'] = scores_pathologies['UP endplate'] + scores_pathologies['LOW endplate']
    # Drop UP and LOW Endplate columns
    #scores_pathologies = scores_pathologies.drop(columns=['UP endplate', 'LOW endplate'])
    scores_pathologies.rename(columns={grading_df.columns[-1]:"Disc Degeneration", "Modic": "Modic Changes"}, inplace=True)
    grading_df.rename(columns={grading_df.columns[-1]:"Disc Degeneration", "Modic": "Modic Changes"}, inplace=True)
    pathologies = grading_df.columns[1:]  # Exclude 'pid'
    statistical_test={}
    for i, pathology in enumerate(pathologies):
        plot_data = []
        statistical_test[pathology] ={}

        for structure in spine_structures:
            th = 10 if pathology == 'Disc Degeneration' else 0
            group1 = scores_pathologies[scores_pathologies[pathology] > th][structure]
            group2 = scores_pathologies[scores_pathologies[pathology] <= th][structure]

            # Perform independent t-test
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            statistical_test[pathology].update({structure:[t_stat, p_value]})

            # Append data for each structure
            plot_data.append(pd.DataFrame({
                'DICE': pd.concat([group1, group2]),
                f"{pathology}": ['Present'] * len(group1) + ['Absent'] * len(group2),
                'Structure': [structure] * (len(group1) + len(group2))
            }))

        # Concatenate all data
        plot_data = pd.concat(plot_data)        # Plotting
        sns.boxplot(x='Structure', y='DICE', hue=f"{pathology}", data=plot_data, palette='Greys',
                    linewidth=1.5, ax=axes[i])

    # correct the overall p values using the Benjamini-Hochberg procedure
    p_values = [v[-1] for s in statistical_test.values() for v in s.values()]
    reject_H0, corrected_p_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[:2]

    corrected_statistical_test = {}

    for i, pathology in enumerate(pathologies):
        corrected_statistical_test[pathology] = {}
        for j, s in enumerate(spine_structures):
            corrected_statistical_test[pathology][s] = statistical_test[pathology][s][:1] + [corrected_p_values[i*len(spine_structures) + j]]

    statistical_test['corrected_p_values'] = corrected_p_values
    for i, pathology in enumerate(pathologies):
        for j, s in enumerate(spine_structures):
            # Print statistical test results in each subplot
            print("Pathology: ", pathology, f"Corrected {s} p-value: {statistical_test[pathology][s][-1]:.3e}")

            axes[i].text(-0.25 , plot_data['DICE'].min() + j*0.015,
                         f"{s} {statistical_test[pathology][s][0]:.2f} ({statistical_test[pathology][s][-1]:.2e})",
                         ha='left', va='bottom', color='black',
                         fontsize=8)

        axes[i].text(-0.25, plot_data['DICE'].min() + j * 0.0215,
                     f"T-test statistic (p-value)",  ha='left', va='bottom',
                     color='black', fontweight='bold', fontsize=10)
        axes[i].set_title(f'{pathology.title()}')
        axes[i].set_ylabel('Dice Score')
        axes[i].set_xlabel('Spinal Structure')
        locs, labels = plt.xticks()
        axes[i].legend(title=f"{pathology}".title(), loc='lower right')
        #axes[i].set_xticks(rotation=45, ha='right',ticks=np.arange(len(spine_structures)),  labels=spine_structures)


    plt.suptitle(title if title is not None else 'Segmentation Performance vs Pathology Analysis',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(Path(save_path)/f'performance_pathology_analysis_corrected.pdf')
