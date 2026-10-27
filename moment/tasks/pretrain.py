import os
import subprocess
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from wandb import AlertLevel

from moment.common import PATHS
from moment.models.moment import MOMENT
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists

from .base import Tasks

warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args

    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp and self.device.type == "cuda",
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                auxiliary_loss = getattr(outputs, "aux_loss", None)
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss

                losses.append(loss.item())

                if return_preds:
                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(outputs.reconstruction.detach().cpu().numpy())
                    masks.append(outputs.pretrain_mask.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)
            return average_loss, losses, (trues, preds, masks)
        else:
            return average_loss

    def train(self):
        self.run_name = self.logger.name
        path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp and self.device.type == "cuda")
        self._init_lr_scheduler()

        self.model.to(self.device)
        # self.evaluate_and_log()

        opt_steps = 0
        cur_epoch = 0
        if len(self.train_dataloader) == 0:
            raise ValueError("Training dataloader is empty")
        while opt_steps < self.args.max_opt_steps and cur_epoch < self.args.max_epoch:
            self.model.train()

            for batch_x in tqdm(
                self.train_dataloader, total=len(self.train_dataloader)
            ):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp and self.device.type == "cuda",
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                reconstruction_loss = loss
                auxiliary_loss = getattr(outputs, "aux_loss", None)
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss

                training_metrics = {
                    "step_train_loss": loss.item(),
                    "reconstruction_loss": reconstruction_loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                for name in ("vq_loss", "moe_aux_loss"):
                    diagnostic = getattr(outputs, name, None)
                    if diagnostic is not None:
                        training_metrics[name] = diagnostic.item()
                self.logger.log(training_metrics)

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(loss, outputs, batch_x)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                if opt_steps % self.args.log_interval == 0:
                    self.evaluate_and_log()

                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )
                    self.save_model(
                        self.model, path, opt_steps, self.optimizer, self.scaler
                    )
                    if getattr(self.args, "external_evaluation", False):
                        self.evaluate_model_external(path, opt_steps, self.device)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                if opt_steps >= self.args.max_opt_steps:
                    break

            cur_epoch = cur_epoch + 1

        if getattr(self.model, "vq_vae", None) is not None:
            self.build_memory_bank(path)
        self.save_model(self.model, path, opt_steps, self.optimizer, self.scaler)
        return self.model

    @torch.no_grad()
    def build_memory_bank(self, path):
        """Collect final-model codes from all training examples, without masking."""
        bank_loader = torch.utils.data.DataLoader(
            self.train_dataloader.dataset, batch_size=self.args.val_batch_size,
            shuffle=False, drop_last=False, num_workers=0,
            collate_fn=self.train_dataloader.collate_fn,
        )
        was_training = self.model.training
        self.model.eval()
        self.model.vq_vae.begin_memory_collection()
        try:
            for batch in bank_loader:
                self.model.reconstruct(
                    x_enc=batch.timeseries.float().to(self.device),
                    input_mask=batch.input_mask.long().to(self.device),
                )
        finally:
            self.model.vq_vae.end_memory_collection()
            self.model.train(was_training)
        self.model.vq_vae.export_memory(os.path.join(path, "codebook_memory.pt"))

    def evaluate_model(self):
        return MetricsStore(val_loss=self.validation(self.val_dataloader))

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log({"validation_loss": eval_metrics.val_loss})
        return eval_metrics

    def evaluate_model_external(self, path: str, opt_steps: int, device) -> None:
        if device.type != "cuda" or torch.cuda.device_count() <= device.index + 1:
            warnings.warn("External evaluation skipped: no additional CUDA device is available")
            return
        print("starting evaluation")
        eval_device = int(str(device).split(":")[-1]) + 1
        command = [
            "python",
            "scripts/evaluation/evaluation.py",
            f"--checkpoint_path={path}",
            f"--opt_steps={opt_steps}",
            f"--run_name={self.run_name}",
            f"--gpu_id={eval_device}",
        ]
        outfile = open(f"{path}/eval_output.txt", "w")
        errfile = open(f"{path}/eval_error.txt", "w")
        subprocess.Popen(command, stdout=outfile, stderr=errfile)
