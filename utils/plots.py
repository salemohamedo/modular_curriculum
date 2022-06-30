import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.utils import load_df



def plot_perf(df, save_path=None, **plot_args):
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    def make_plot(df, ax):
        sns.lineplot(data=df, x='n_op', y='value', ax=ax, **plot_args)

    df.loc[df['metric'] == 'acc', 'value'] = 1 - df['value'] ## change accuracies to errors
    df.loc[df['metric'] == 'acc', 'metric'] = 'error'

    for i, d in enumerate([('iid', 'ID'), ('ood', 'OOD')]):
        subfigs[i].suptitle(d[1], fontweight="bold")
        axes = subfigs[i].subplots(1, 2)
        for j, t in enumerate(['Regression', 'Classification']):
            metric = 'error' if t == 'Classification' else 'loss'
            make_plot(
                df.loc[(df['type'] == t.lower()) 
                & (df['dist'] == d[0]) 
                & (df['metric'] == metric)], 
                axes[j])
            axes[j].set_ylabel(metric)
            axes[j].set_xlabel('rules')
            axes[j].set_title(t)
            axes[j].legend(loc=4, title='Curriculum Type')
    if save_path:
        fig.savefig(save_path)

def plot_train_convergence(df, save_path=None, **plot_args):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, task in enumerate(['Regression', 'Classification']):
        df_task = df.loc[(df['type'] == task.lower())]
        sns.lineplot(x='iteration', y='value', data=df_task, ax=axes[i], **plot_args)
        axes[i].set_ylabel('loss')
        axes[i].set_title(task)
    # df_class = df.loc[(df['type'] == 'classification') & (df['seq_len'] == 10) & (df['metric'] == 'loss')]
    # sns.lineplot(x='iteration', y='value', data=df_class, ax=axes[1], **plot_args)
    # axes[1].set_ylabel('loss')
    # axes[1].set_title('Classification')
    if save_path:
        fig.savefig(save_path)

def plot_spec_metric(df, y, ylabel, save_path=None, **plot_args):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # lss = [':', '-', ':', '-']
    for i, setting in enumerate(['mod', 'mod_dich']):
        for j, task in enumerate(['Regression', 'Classification']):
            df_task = df.loc[(df['type'] == task.lower()) &
                             (df['setting'] == setting.lower())]
            sns.lineplot(x='n_op', y=y, data=df_task, ax=axes[i][j], **plot_args)
            axes[i][j].set_ylabel(ylabel)
            axes[i][j].set_xlabel('# Rules')
            if i == 0:
                axes[i][j].set_title(task)
            # handles = axes[i].legend_.legendHandles[::-1]
            # for line, ls, handle in zip(axes[i].lines, lss, handles):
            #     line.set_linestyle(ls)
            #     handle.set_ls(ls)
    if save_path:
        fig.savefig(save_path)
    return fig, axes


def plot_spec_metric_2(df, y, ylabel, save_path=None, **plot_args):
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    for i, setting in enumerate([['mod', 'Modular'], ['mod_dich', 'Mod Dichotomic']]):
        subfigs[i].suptitle(setting[1], fontweight="bold")
        axes = subfigs[i].subplots(1, 2)
        for j, task in enumerate(['Regression', 'Classification']):
            df_task = df.loc[(df['type'] == task.lower()) &
                             (df['setting'] == setting[0].lower())]
            sns.lineplot(x='n_op', y=y, data=df_task, ax=axes[j], **plot_args)
            axes[j].set_ylabel(ylabel)
            axes[j].set_xlabel('# Rules')
            axes[j].set_title(task)
            axes[j].legend(loc=4, title='Curriculum Type')
    if save_path:
        fig.savefig(save_path)
    return fig, axes

# def plot_collapse(df, worst=False, save_path=None, **plot_args):
#     if worst:
#         y = 'collapse_worst'
#         ylabel = 'Collapse Worst'
#     else:
#         y = 'collapse_avg'
#         ylabel = 'Collapse Avg'

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     df_reg = df.loc[df['type'] == 'regression']
#     sns.barplot(x='n_op', y=y, data=df_reg, ax=axes[0], **plot_args)
#     axes[0].set_ylabel(ylabel)
#     axes[0].set_xlabel('# Rules')
#     axes[0].set_title('Regression')

#     df_class = df.loc[df['type'] == 'classification']
#     sns.barplot(x='n_op', y=y, data=df_class, ax=axes[1], **plot_args)
#     axes[1].set_ylabel(ylabel)
#     axes[1].set_xlabel('# Rules')
#     axes[1].set_title('Classification')
#     if save_path:
#         fig.savefig(save_path)
