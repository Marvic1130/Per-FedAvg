# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""

import torch
import numpy as np
import random
from client import train, test, local_adaptation
from model import ANN
import copy
from tqdm import tqdm
import wandb
import time
from src.data.leaf_loader import load_leaf_datasets, load_leaf_splits_datasets, auto_select_leaf_splits
from src.utils.log_helpers import build_round_log
from pathlib import Path
import pandas as pd


# Implementation for per-fedavg server.
class PerFed:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        
        # Apply cfg defaults if not present in args
        if self.cfg:
            if not hasattr(self.args, 'r'): self.args.r = self.cfg.num_rounds
            if not hasattr(self.args, 'local_epochs'): self.args.local_epochs = self.cfg.client.local_epochs
            if not hasattr(self.args, 'B'): self.args.B = self.cfg.client.batch_size
            if not hasattr(self.args, 'lr'): self.args.lr = self.cfg.server.inner_lr
            if not hasattr(self.args, 'alpha'): self.args.alpha = 0.01 # Default alpha
            if not hasattr(self.args, 'beta'): self.args.beta = 0.001 # Default beta
            if not hasattr(self.args, 'C'): self.args.C = 1.0 # Default sampling rate
            
            # Load datasets via unified routing
            train_files = getattr(self.cfg.dataset, 'train_json_files', None) or getattr(self.cfg.dataset, 'json_files', None)
            val_files = getattr(self.cfg.dataset, 'val_json_files', None)
            holdout_files = getattr(self.cfg.dataset, 'holdout_json_files', None)
            if train_files:
                # Use unified splits loader with global round-robin limit to ensure identical client cohorts
                self.train_datasets, self.test_datasets, self.holdout_datasets = load_leaf_splits_datasets(
                    root=self.cfg.dataset.root,
                    train_files=train_files,
                    val_files=val_files,
                    holdout_files=holdout_files,
                    limit=self.cfg.dataset.num_clients,
                )
            else:
                # Auto selection across all JSONs with deterministic sampling
                splits = auto_select_leaf_splits(
                    root=self.cfg.dataset.root,
                    batch_size=self.cfg.client.batch_size,
                    shuffle=True,
                    num_clients=int(getattr(self.cfg.dataset, 'num_clients', 0) or 0),
                    seed=int(getattr(self.cfg, 'seed', 42)),
                    holdout_limit=int(getattr(self.cfg.dataset, 'holdout_client_limit', 0) or 0) or None,
                )
                # Extract TensorDatasets from DataLoaders
                self.train_datasets = {cid: ldr.dataset for cid, ldr in splits.train.items()}
                self.test_datasets = {cid: ldr.dataset for cid, ldr in splits.val.items()}
                self.holdout_datasets = splits.holdout
            if self.holdout_datasets:
                print(f"Loaded {len(self.holdout_datasets)} holdout clients")

            self.client_ids = list(self.train_datasets.keys())
            self.args.K = len(self.client_ids)
            self.args.clients = self.client_ids

            # Initialize wandb if not already initialized (it might be by the pipeline)
            # But here we might want to ensure it's active or re-init?
            # The pipeline orchestrator doesn't init wandb for baselines usually?
            # Actually, HyperQLoRA pipeline inits wandb.
            # If we run via ExtAdapter, we are part of the pipeline.
            # If we run standalone, we need init.
            # Let's assume wandb is active if cfg is passed.
            # Initialize wandb if not already initialized
            if wandb.run is None:
                wandb.init(
                    project=self.cfg.wandb_project if self.cfg else "HyperQLoRA-HFL",
                    name=f"perfedavg_{self.cfg.experiment_name}" if self.cfg else "perfedavg",
                    config=vars(self.args),
                    reinit=True
                )
        else:
            self.train_datasets = None
            self.test_datasets = None
            self.client_ids = self.args.clients

        # Model initialization needs to be generic
        # ANN is hardcoded for wind dataset.
        # We need to use the model from cfg or args.
        from src.client.models import get_model_class
        ModelClass = get_model_class(self.cfg.dataset.name) if self.cfg else None
        
        if ModelClass:
            dname = str(self.cfg.dataset.name).lower()
            feat_dim = getattr(self.cfg.dataset, "feature_dim", None)
            if dname in {"sent140", "shakespeare"}:
                vocab_size = int(feat_dim or 10000)
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes, vocab_size=vocab_size).to(args.device)
            elif dname == "synthetic":
                input_dim = int(feat_dim or 60)
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes, input_dim=input_dim).to(args.device)
            elif dname == "extrasensory":
                input_dim = int(feat_dim or 226)
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes, input_dim=input_dim).to(args.device)
            elif dname == "har_lstm":
                input_size = int(feat_dim or 3)
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes, input_size=input_size).to(args.device)
            elif dname in {"har", "har_mlp"}:
                input_dim = int(feat_dim or 47)
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes, input_dim=input_dim).to(args.device)
            else:
                self.nn = ModelClass(num_classes=self.cfg.dataset.num_classes).to(args.device)
        elif self.cfg:
            # Use a generic model or the one specified in cfg
            # For now, let's assume CNN for FEMNIST
            from models import CNN  # Import from FedAvg models? Or define here?
            # Per-FedAvg has its own model.py with ANN.
            # We should probably use the same model class as FedAvg for consistency on FEMNIST.
            # But Per-FedAvg expects specific model structure?
            # Let's try to import CNN from FedAvg's models if possible, or define a simple one.
            # Or use the one from pFedMe/models?
            # Let's use the one from FedAvg/models since we are in the same repo context.
            import sys

            sys.path.append("external/FedAvg")
            from models import CNN

            self.nn = CNN(n_channels=1, n_classes=self.cfg.dataset.num_classes).to(args.device)
        else:
            self.nn = ANN(args=self.args, name='server').to(args.device)

        self.nns = []
        # init
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            # temp.name = self.args.clients[i] # CNN doesn't have name attr usually
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.r), desc='round'):
            start_time = time.time()
            # print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch parameters
            self.dispatch(index)
            # local updating
            client_durations = self.client_update(index, t)
            
            server_start_time = time.time()
            # aggregation parameters
            self.aggregation(index)
            server_duration = time.time() - server_start_time

            duration = time.time() - start_time
            
            max_client_duration = max(client_durations) if client_durations else 0.0
            mean_client_duration = np.mean(client_durations) if client_durations else 0.0
            simulated_duration = max_client_duration + server_duration

            # Evaluation and Logging
            if self.cfg:
                wandb.log({
                    "duration": duration,
                    "client_duration": mean_client_duration,
                    "max_client_duration": max_client_duration,
                    "server_duration": server_duration,
                    "simulated_duration": simulated_duration
                }, step=t, commit=False)
                
                self.evaluate(t)

        # After training rounds, export final per-client test metrics
        try:
            final_rows = []
            loss_fn = torch.nn.CrossEntropyLoss().to(self.args.device)
            for i, cid in enumerate(self.client_ids):
                if cid not in self.test_datasets:
                    continue
                dataset = self.test_datasets[cid]
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.B, shuffle=False)
                # Personalized adaptation
                model = copy.deepcopy(self.nn)
                model = local_adaptation(self.args, model, self.train_datasets[cid])
                model.eval()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(self.args.device), target.to(self.args.device)
                        out = model(data)
                        loss = loss_fn(out, target)
                        total_loss += loss.item() * data.size(0)
                        pred = out.argmax(dim=1)
                        total_correct += (pred == target).sum().item()
                        total_samples += data.size(0)
                if total_samples > 0:
                    final_rows.append({
                        'client_id': cid,
                        'num_samples': total_samples,
                        'test_loss': total_loss / total_samples,
                        'test_acc': total_correct / total_samples,
                    })
            if final_rows:
                df = pd.DataFrame(final_rows)
                out_dir = Path('save/perfedavg')
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / "final.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved Per-FedAvg final per-client metrics to {csv_path}")
        except Exception as e:
            print(f"[Per-FedAvg] Failed to save final per-client metrics: {e}")

        return self.nn

    def evaluate(self, round_idx):
        # Evaluate global model (or personalized models?)
        # Per-FedAvg is about personalized models.
        # We should evaluate personalized models on test set.

        total_test_loss = 0.0
        total_test_acc = 0.0  # We need to implement acc calculation in test()
        total_samples = 0
        client_losses = []
        client_accs = []

        # For efficiency, maybe evaluate on a subset or all?
        # Let's evaluate on all clients for now (or sampled ones).

        # We need to modify test() in client.py to return loss and acc.
        # Currently it returns nothing (just prints?).

        # Let's modify client.py's test function first?
        # Or implement evaluation here.

        # Let's implement a simple evaluation loop here using the client's test data

        loss_fn = torch.nn.CrossEntropyLoss().to(self.args.device)

        for i in range(self.args.K):
            cid = self.client_ids[i]
            if cid not in self.test_datasets:
                continue

            dataset = self.test_datasets[cid]
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.B, shuffle=False)

            # Personalize model for client
            model = copy.deepcopy(self.nn)
            # In Per-FedAvg, self.nns[i] holds the personalized parameters?
            # No, self.nns[i] is updated in client_update.
            # But aggregation overwrites self.nn.
            # Per-FedAvg maintains one model per client?
            # In this implementation:
            # dispatch: self.nns[j] <- self.nn
            # client_update: self.nns[k] updated
            # aggregation: self.nn <- average(self.nns)

            # So self.nns[i] is the personalized model AFTER update?
            # But dispatch overwrites it at start of round.
            # So at end of round, self.nns[i] is the updated local model.

            # To evaluate personalized performance, we should use self.nns[i] (after update)
            # OR perform local adaptation on global model (Meta-Learning style).
            # The paper says: "Per-FedAvg aims to find an initial shared model that users can easily adapt"
            # So we should take self.nn, adapt it on train data, then evaluate on test data.

            model = copy.deepcopy(self.nn)
            # Local adaptation
            model = local_adaptation(self.args, model, self.train_datasets[cid])
            model.eval()

            correct = 0
            loss_sum = 0.0
            samples = 0

            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss_sum += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    samples += data.size(0)

            if samples > 0:
                total_test_loss += loss_sum
                total_test_acc += correct
                total_samples += samples
                client_losses.append(loss_sum / samples)
                client_accs.append(correct / samples)

        avg_test_loss = total_test_loss / total_samples if total_samples > 0 else 0.0
        avg_test_acc = total_test_acc / total_samples if total_samples > 0 else 0.0

        log_data = build_round_log(
            round_idx=round_idx + 1,
            meta_loss=avg_test_loss,
            client_losses=client_losses,
            client_accs=client_accs,
            val_losses=client_losses,
            val_accs=client_accs,
        )
        wandb.log(log_data, step=round_idx)

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data / len(index)
                # params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index, t):  # update nn
        client_durations = []
        for k in index:
            dataset = self.train_datasets[self.client_ids[k]] if self.train_datasets else None
            client_start_time = time.time()
            self.nns[k] = train(self.args, self.nns[k], k, t, dataset=dataset)
            client_durations.append(time.time() - client_start_time)
        return client_durations

    def evaluate_holdout(self):
        if not self.holdout_datasets:
            return
            
        print("Evaluating on holdout data...")
        self.nn.eval()
        
        rows = []
        losses = []
        accs = []
        total_samples = 0
        total_loss = 0.0
        total_acc = 0.0
        
        loss_fn = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        with torch.no_grad():
            for cid, dataset in self.holdout_datasets.items():
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.B, shuffle=False)
                
                client_loss = 0.0
                client_correct = 0.0
                client_samples = 0
                
                for data, target in loader:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    output = self.nn(data)
                    loss = loss_fn(output, target)
                    
                    client_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    client_correct += pred.eq(target.view_as(pred)).sum().item()
                    client_samples += data.size(0)
                
                if client_samples > 0:
                    avg_loss = client_loss / client_samples
                    avg_acc = client_correct / client_samples
                    
                    rows.append((cid, client_samples, avg_loss, avg_acc))
                    losses.append(avg_loss)
                    accs.append(avg_acc)
                    
                    total_loss += client_loss
                    total_acc += client_correct
                    total_samples += client_samples
        
        if not rows:
            return
            
        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_acc / max(total_samples, 1)
        
        log_data = {
            "holdout/loss_mean": mean_loss,
            "holdout/loss_std": float(np.std(losses)) if len(losses) > 1 else 0.0,
            "holdout/loss_min": min(losses),
            "holdout/loss_max": max(losses),
            "holdout/acc_mean": mean_acc,
            "holdout/acc_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
            "holdout/acc_min": min(accs),
            "holdout/acc_max": max(accs),
            "holdout/num_clients": len(rows),
            "holdout/num_samples": total_samples,
        }
        
        if wandb.run is not None:
            table = wandb.Table(columns=["client_id", "samples", "loss", "accuracy"])
            for row in rows:
                table.add_data(*row)
            log_data["holdout/client_metrics"] = table
            wandb.log(log_data)
            
        print(f"Holdout Evaluation: Loss={mean_loss:.4f}, Acc={mean_acc:.4f} (Clients={len(rows)})")
        # Save holdout per-client metrics to CSV
        try:
            out_dir = Path("save")
            out_dir.mkdir(parents=True, exist_ok=True)
            client_rows = []
            for cid, samples, loss, acc in rows:
                client_rows.append({
                    "client_id": cid,
                    "num_samples": int(samples),
                    "holdout_loss": float(loss),
                    "holdout_acc": float(acc),
                })
            if client_rows:
                df = pd.DataFrame(client_rows)
                out_dir = Path('save/perfedavg')
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / "holdout.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved Per-FedAvg holdout per-client metrics to {csv_path}")
        except Exception as e:
            print(f"[Per-FedAvg] Failed to save holdout per-client metrics: {e}")

    def global_test(self):
        self.evaluate_holdout()
