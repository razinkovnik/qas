from dataclasses import dataclass


@dataclass
class TrainingArguments:
    output_dir = "models"
    log_dir = "runs"
    evaluate_during_training = True
    train_batch_size = 2
    eval_batch_size = 4
    block_size = 256
    learning_rate = 5e-5
    max_grad_norm = 1.0
    save_steps = 10
    device = "cuda"
