import os
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np

class RegressionTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, device, is_ddp=False, rank=0):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.is_ddp = is_ddp
        self.rank = rank
        
        self.epochs = config["train"]["epochs"]
        self.lr = config["train"]["lr"]
        self.weight_decay = config["train"]["weight_decay"]
        self.grad_accum_steps = config["train"]["grad_accum_steps"]
        self.use_amp = bool(config["train"]["amp"] and device.type == "cuda")
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        self.criterion = nn.L1Loss() # standard robust loss for 3D pose regression
        
        self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler(self.lr)
        
    def _setup_optimizer_and_scheduler(self, lr):
        # Use fused AdamW for faster execution if available on CUDA
        use_fused = self.device.type == "cuda"
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=self.weight_decay,
            fused=use_fused
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return optimizer, scheduler

    def train(self, checkpoint_dir):
        if self.rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        best_val_loss = float('inf')
        
        # Setup DataLoaders
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.is_ddp else None
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.is_ddp else None
        
        train_loader = self._get_dataloader(self.train_dataset, self.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
        val_loader = self._get_dataloader(self.val_dataset, self.batch_size, shuffle=False, sampler=val_sampler)
        
        for epoch in range(1, self.epochs + 1):
            if self.is_ddp:
                train_sampler.set_epoch(epoch)
                
            success = False
            while not success:
                try:
                    epoch_start = time.time()
                    train_loss, samples_per_sec = self._train_one_epoch(train_loader)
                    success = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.batch_size > 2:
                        if self.rank == 0:
                            print(f"\n[WARNING] CUDA Out of Memory with batch size {self.batch_size}. Reducing batch size to {self.batch_size // 2}...")
                        torch.cuda.empty_cache()
                        self.batch_size = self.batch_size // 2
                        self.grad_accum_steps = self.grad_accum_steps * 2
                        train_loader = self._get_dataloader(self.train_dataset, self.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
                        val_loader = self._get_dataloader(self.val_dataset, self.batch_size, shuffle=False, sampler=val_sampler)
                    else:
                        raise e
                        
            val_loss = self._validate(val_loader)
            self.scheduler.step()
            
            if self.rank == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                      f"Speed: {samples_per_sec:.1f} samples/s | Time: {epoch_time:.1f}s | Batch: {self.batch_size}")
                
                # Save checkpoints
                # Unwrap DDP model for saving
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_loss": val_loss,
                    "config": self.config
                }
                
                torch.save(checkpoint, os.path.join(checkpoint_dir, "last.pth"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pth"))
                    
        # Synchronize all processes before returning
        if self.is_ddp:
            dist.barrier()
            
        return best_val_loss

    def _get_dataloader(self, dataset, batch_size, shuffle=False, sampler=None):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True if len(dataset) >= batch_size else False,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def _train_one_epoch(self, dataloader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        num_samples = 0
        
        start_time = time.time()
        
        for i, (csi, pose, metadata) in enumerate(dataloader):
            # Channels Last memory format optimization for CNNs
            csi = csi.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            pose = pose.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred_pose = self.model(csi)
                loss = self.criterion(pred_pose, pose)
                loss = loss / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            # Unscale before clipping
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping to prevent exploding gradients and NaN
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Check for NaN in loss
                if not torch.isnan(loss):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.rank == 0:
                        print(f"[WARNING] NaN loss detected at step {i}, skipping optimizer step.")
                        
                self.optimizer.zero_grad(set_to_none=True)
                
            running_loss += loss.item() * self.grad_accum_steps
            num_samples += csi.size(0)
            
        end_time = time.time()
        samples_per_sec = num_samples / (end_time - start_time) if end_time > start_time else 0
        
        # In DDP, we should average the loss across all GPUs
        avg_loss = running_loss / len(dataloader)
        if self.is_ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / dist.get_world_size()).item()
            
            # Aggregate samples/sec for accurate reporting
            sps_tensor = torch.tensor(samples_per_sec, device=self.device)
            dist.all_reduce(sps_tensor, op=dist.ReduceOp.SUM)
            samples_per_sec = sps_tensor.item()
            
        return avg_loss, samples_per_sec

    def _validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for csi, pose, metadata in dataloader:
                csi = csi.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                pose = pose.to(self.device, non_blocking=True)
                
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred_pose = self.model(csi)
                    loss = self.criterion(pred_pose, pose)
                    
                running_loss += loss.item()
                
        avg_loss = running_loss / len(dataloader)
        if self.is_ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / dist.get_world_size()).item()
            
        return avg_loss
