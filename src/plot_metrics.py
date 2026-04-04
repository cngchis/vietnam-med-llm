import os
import matplotlib.pyplot as plt

def plot_training_metrics(trainer, output_dir):
    logs = trainer.state.log_history

    train_steps = []
    train_loss = []

    eval_steps = []
    eval_loss = []

    # ====== Extract data ======
    for log in logs:
        if "loss" in log and "eval_loss" not in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])

        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_loss.append(log["eval_loss"])

    # ====== Plot TRAIN LOSS ======
    plt.figure()
    plt.plot(train_steps, train_loss)
    plt.xlabel("Steps")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    # ====== Plot EVAL LOSS ======
    plt.figure()
    plt.plot(eval_steps, eval_loss)
    plt.xlabel("Steps")
    plt.ylabel("Eval Loss")
    plt.title("Validation Loss")
    plt.savefig(os.path.join(output_dir, "eval_loss.png"))
    plt.close()

    # ====== Plot BOTH ======
    plt.figure()
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.plot(eval_steps, eval_loss, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_vs_eval.png"))
    plt.close()

    print("Saved plots to", output_dir)