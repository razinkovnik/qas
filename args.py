from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


@dataclass
class TrainingArguments:
    output_dir = "models"
    _writer: Optional[SummaryWriter] = field(default=None)
    train_batch_size = 8
    test_batch_size = 8
    block_size = 128
    learning_rate = 5e-5
    max_grad_norm = 1.0
    save_steps = 50
    device = "cpu"
    model_name = "DeepPavlov/rubert-base-cased"
    train_dataset = ""
    test_dataset = ""
    num_train_epochs = 1
    load = False

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @writer.setter
    def writer(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)
