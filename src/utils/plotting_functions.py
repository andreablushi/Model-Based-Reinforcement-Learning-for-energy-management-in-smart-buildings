import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from google.colab import files  # Google Colab specific module for file downloads

# Suppress warnings to avoid cluttering output
warnings.filterwarnings("ignore")

# Set global plotting style using seaborn
sns.set_theme(style='darkgrid', context='paper')

def save_or_download(fig, save_path, save=False):
    """
    Handles figure display/saving for Google Colab environments.
    
    Args:
        fig (matplotlib.figure.Figure): Figure object to process
        save_path (str): File path for saving (must end with '.svg')
        save (bool): If True, saves as SVG and downloads file. If False, displays inline.
    """
    if save:
        fig.savefig(save_path, format='svg', bbox_inches='tight')  # Save as vector graphic
        files.download(save_path)  # Trigger download (Colab-specific)
        plt.close(fig)  # Free memory
    else:
        plt.show()  # Standard inline display
        plt.close(fig)

def plot_comparison(agent_data1, agent_data2, agent_data3,
                    ylabel, title,
                    save=False,
                    save_path="plot_comparison.svg"):
    """
    Plots comparison of three agents' time-series data.
    
    Args:
        agent_data1/2/3 (list/array): Data series for each agent
        ylabel (str): Y-axis label text
        title (str): Plot title
        save (bool): Save/download control
        save_path (str): Output file path
    """
    steps = range(len(agent_data1))
    fig, ax = plt.subplots(figsize=(20, 5))
    colors = sns.color_palette(n_colors=3)
    
    # Plot each agent's data with distinct colors
    ax.plot(steps, agent_data1, label='BasicRBC', color=colors[0], alpha=0.6)
    ax.plot(steps, agent_data2, label='OptimizedRBC', color=colors[1], alpha=0.6)
    ax.plot(steps, agent_data3, label='BasicBatteryRBC', color=colors[2], alpha=0.6)
    
    # Configure plot elements
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(steps[::48])  # Show every 48th step 
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_or_download(fig, save_path, save)

def plot_generic_comparison(agent_data1, agent_data2, agent_data3,
                            ylabel, title, labels=None,
                            save=False,
                            save_path="plot_generic_comparison.svg"):
    """
    Flexible three-agent comparison with customizable labels.
    
    Args:
        labels (list): Optional custom legend labels for agents
        Other arguments same as plot_comparison()
    """
    steps = range(len(agent_data1))
    if not labels or len(labels) != 3:
        labels = ['Agente 1', 'Agente 2', 'Agente 3']  # Default labels
    
    fig, ax = plt.subplots(figsize=(20, 5))
    colors = sns.color_palette(n_colors=3)
    
    ax.plot(steps, agent_data1, label=labels[0], color=colors[0], alpha=0.6)
    ax.plot(steps, agent_data2, label=labels[1], color=colors[1], alpha=0.6)
    ax.plot(steps, agent_data3, label=labels[2], color=colors[2], alpha=0.6)
    
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(steps[::48])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_or_download(fig, save_path, save)

def plot_single_agent(agent_data, ylabel, title,
                      label='Agent',
                      save=False,
                      save_path="plot_single_agent.svg"):
    """
    Plots data for a single agent.
    
    Args:
        agent_data (list/array): Time-series data for one agent
        label (str): Legend label for the agent
        Other arguments consistent with other plotting functions
    """
    steps = range(len(agent_data))
    fig, ax = plt.subplots(figsize=(20, 5))
    
    ax.plot(steps, agent_data, label=label, color='tab:red', alpha=0.9)
    
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(steps[::48])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_or_download(fig, save_path, save)

def plot_comparison_two(agent_data1, agent_data2,
                        ylabel, title,
                        label1='Agent 1', label2='Agent 2',
                        save=False,
                        save_path="plot_comparison_two.svg"):
    """
    Compares two agents' time-series data.
    
    Args:
        label1/label2 (str): Custom legend labels
        Other parameters consistent with three-agent comparison
    """
    steps = range(len(agent_data1))
    fig, ax = plt.subplots(figsize=(20, 5))
    colors = sns.color_palette(n_colors=2)
    
    ax.plot(steps, agent_data1, label=label1, color=colors[0], alpha=0.8)
    ax.plot(steps, agent_data2, label=label2, color=colors[1], alpha=0.8)
    
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(steps[::48])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_or_download(fig, save_path, save)

def plot_rewards(agent_data,
                 ylabel='Reward', title='Reward per Agente',
                 label='Ricompense',
                 save=False,
                 save_path="plot_rewards.svg"):
    """
    Specialized plot for reward metrics with simplified formatting.
    
    Args:
        agent_data (list/array): Reward values over steps
        Uses smaller figure size and shows all steps on x-axis
    """
    steps = range(len(agent_data))
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps, agent_data, label=label, color='tab:orange', alpha=0.9)
    
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(steps)  # Show every step (use with caution for large data)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_or_download(fig, save_path, save)

def create_episode_table(episode_data):
    """
    Creates a DataFrame summarizing episode reward statistics.
    
    Args:
        episode_data (list): List of dictionaries with reward metrics per episode
    
    Returns:
        pd.DataFrame: Formatted table with columns:
            ['Episode', 'Reward Min', 'Reward Max', 'Reward Sum', 'Reward Average']
    """
    episode_numbers = list(range(1, len(episode_data) + 1))
    min_rewards = [ep['min'][0] for ep in episode_data]
    max_rewards = [ep['max'][0] for ep in episode_data]
    sum_rewards = [ep['sum'][0] for ep in episode_data]
    mean_rewards = [ep['mean'][0] for ep in episode_data]
    
    df = pd.DataFrame({
        'Episodio': episode_numbers,
        'Reward Min': min_rewards,
        'Reward Max': max_rewards,
        'Reward Somma': sum_rewards,
        'Reward Media': mean_rewards
    })
    return df

def plot_learning_metrics(data_dir, step_filter=0,
                          save=False,
                          save_dir="plots"):
    """
    Automates plotting of training/evaluation metrics from log files.
    
    Args:
        data_dir (str): Directory containing log CSV files
        step_filter (int): Minimum step value to include (filters early training)
        save (bool): Save/download control
        save_dir (str): Directory to save output plots
    """
    def load_df(filename):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            return None
            
        # Check if file is empty
        if os.path.getsize(path) == 0:
            print(f"Warning: Empty file skipped - {path}")
            return None
            
        try:
            df = pd.read_csv(path)
            # Handle empty DataFrames (no rows)
            if df.empty:
                print(f"Warning: Empty DataFrame - {path}")
                return None
                
            if 'step' in df.columns:
                df = df[df['step'] >= step_filter]
            return df
            
        except pd.errors.EmptyDataError:
            print(f"Warning: EmptyDataError - {path}")
            return None
        except Exception as e:
            print(f"Error reading {path}: {str(e)}")
            return None


    # File configuration: (filename, [metrics to plot])
    files = [
        ('model_train.csv', ['model_loss', 'model_val_score']),
        ('train.csv', ['actor_loss', 'critic_loss', 'batch_reward']),
        ('results.csv', None),  # None = plot all columns
        ('eval.csv', None)
    ]
    
    if save:
        os.makedirs(save_dir, exist_ok=True)  # Create output dir if needed

    # Process each configured file
    for filename, metrics in files:
        df = load_df(os.path.join(data_dir, filename))
        if df is None or df.empty:
            continue
            
        base = os.path.splitext(filename)[0]
        
        # Special handling for model_train.csv (dual-axis plot)
        if filename == 'model_train.csv':
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax2 = ax1.twinx()
            
            if 'model_loss' in df:
                ax1.plot(df['step'], df['model_loss'], label='Model Loss', color='red')
                ax1.set_ylabel('Model Loss')
                
            if 'model_val_score' in df:
                ax2.plot(df['step'], df['model_val_score'], label='Validation Score')
                ax2.set_ylabel('Validation Score')
                
            ax1.set_xlabel('Step')
            ax1.set_title('Model Metrics')
            lines, labels = ax1.get_legend_handles_labels()
            l2, lbl2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + l2, labels + lbl2, loc='upper left')
            plt.tight_layout()
            save_or_download(fig, os.path.join(save_dir, f"{base}_metrics.svg"), save)
            continue
        
        # Determine columns to plot
        cols = metrics if metrics else [c for c in df.columns if c != 'step']
        
        # Plot each metric column separately
        for col in cols:
            if col not in df:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['step'], df[col], label=col)
            ax.set_xlabel('Step')
            ax.set_ylabel(col)
            ax.set_title(f"{col} Over Steps")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            save_or_download(fig, os.path.join(save_dir, f"{base}_{col}.svg"), save)