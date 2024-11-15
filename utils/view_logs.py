import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def view_logs(run_path=None):
    # Если указан конкретный run_path, загружаем его
    if run_path:
        api = wandb.Api()
        run = api.run(run_path)
        history = pd.DataFrame(run.scan_history())
    else:
        # Иначе показываем список всех ранов
        api = wandb.Api()
        runs = api.runs(str(Path("logs") / "weather-predictions"))
        print("\nДоступные раны:")
        for run in runs:
            print(f"- {run.name}: {run.path}")
        return

    # Построение графиков
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'].unique(), history.groupby('epoch')['train_loss'].last(), label='Train Loss')
    plt.plot(history['epoch'].unique(), history.groupby('epoch')['val_loss'].last(), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # График learning rate
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Run path (e.g., 'username/project/run_id')", default=None)
    args = parser.parse_args()
    
    view_logs(args.run) 