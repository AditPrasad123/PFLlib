import torch
import torch.nn as nn
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from flcore.utils.metrics import FLMetricsTracker


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        
        # Initialize FL metrics tracker
        self.fl_metrics_tracker = FLMetricsTracker()
        
        # Detailed metrics storage
        self.rs_detailed_metrics = []  # Store all detailed metrics per round

        self._register_model_info()

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def _register_model_info(self):
        model_info = self._compute_model_complexity()

        head = None
        for name in ("head", "fc", "classifier"):
            if hasattr(self.global_model, name):
                head = getattr(self.global_model, name)
                break

        if head is not None:
            n_qubits = getattr(head, "n_qubits", None)
            n_layers = getattr(head, "n_layers", None)
            if n_qubits is not None:
                model_info["n_qubits"] = int(n_qubits)
            if n_layers is not None:
                model_info["n_layers"] = int(n_layers)
                model_info["circuit_depth"] = int(n_layers)

        if model_info:
            self.fl_metrics_tracker.set_model_info(model_info)

    def _compute_model_complexity(self):
        """Compute total/trainable param counts and FLOPs via forward hooks (no external packages)."""
        info = {}
        try:
            total = sum(p.numel() for p in self.global_model.parameters())
            trainable = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
            info['total_params'] = total
            info['trainable_params'] = trainable
            info['non_trainable_params'] = total - trainable
        except Exception:
            pass

        try:
            dataset_name = getattr(self, 'dataset', '')
            if '_quanv' in dataset_name:
                dummy_input = torch.zeros(1, 4, 24, 24)
            elif 'MNIST' in dataset_name:
                dummy_input = torch.zeros(1, 1, 28, 28)
            else:
                dummy_input = torch.zeros(1, 3, 224, 224)

            flop_count = [0]
            hooks = []

            def conv_hook(module, inp, out):
                b = inp[0].shape[0]
                c_in = module.in_channels
                c_out = module.out_channels
                kH, kW = (module.kernel_size if isinstance(module.kernel_size, tuple)
                          else (module.kernel_size, module.kernel_size))
                oH, oW = out.shape[2], out.shape[3]
                macs = b * c_out * oH * oW * (c_in // module.groups) * kH * kW
                flop_count[0] += 2 * macs

            def linear_hook(module, inp, out):
                b = inp[0].numel() // module.in_features
                flop_count[0] += 2 * b * module.in_features * module.out_features

            model_cpu = self.global_model.cpu()
            for m in model_cpu.modules():
                if isinstance(m, nn.Conv2d):
                    hooks.append(m.register_forward_hook(conv_hook))
                elif isinstance(m, nn.Linear):
                    hooks.append(m.register_forward_hook(linear_hook))

            model_cpu.eval()
            with torch.no_grad():
                model_cpu(dummy_input)

            for h in hooks:
                h.remove()

            # Move model back to original device
            self.global_model.to(self.device)

            flops = flop_count[0]
            info['flops'] = flops
            if flops >= 1e9:
                info['flops_str'] = f"{flops / 1e9:.3f} GFLOPs"
            elif flops >= 1e6:
                info['flops_str'] = f"{flops / 1e6:.3f} MFLOPs"
            else:
                info['flops_str'] = f"{flops:,} FLOPs"
        except Exception as exc:
            info['flops_str'] = f'N/A ({exc})'

        return info

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
    
    def calculate_communication_cost(self):
        """
        Calculate the communication cost for this round.
        Communication = 2 * (model size from server to clients + model updates from clients to server)
        Returns bytes transferred in this round.
        """
        # Calculate model size in bytes
        total_params = sum(p.numel() for p in self.global_model.parameters())
        # Assuming float32 (4 bytes per parameter)
        model_size_bytes = total_params * 4
        
        # Communication: server sends to all selected clients + clients send updates to server
        # 2 times: download model + upload updates
        communication_cost = 2 * len(self.selected_clients) * model_size_bytes
        
        return communication_cost
        
    def save_results(self):
        # Get model name if available
        model_name = getattr(self.args, 'model_name', 'unknown_model')
        algo = self.dataset + "_" + self.algorithm + "_" + model_name
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                # Save basic metrics
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                
                # Save detailed metrics if available
                if len(self.rs_detailed_metrics) > 0:
                    # Create a group for detailed metrics
                    detailed_group = hf.create_group('detailed_metrics')
                    
                    for round_idx, metrics_dict in enumerate(self.rs_detailed_metrics):
                        round_group = detailed_group.create_group(f'round_{round_idx}')
                        
                        for key, value in metrics_dict.items():
                            try:
                                if isinstance(value, (int, float)):
                                    round_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    round_group.create_dataset(key, data=np.string_(value))
                                elif isinstance(value, np.ndarray):
                                    round_group.create_dataset(key, data=value)
                                elif isinstance(value, dict):
                                    # Save curve dictionaries with their components
                                    if key in ['roc_curve', 'pr_curve']:
                                        if value is not None:
                                            curve_subgroup = round_group.create_group(key)
                                            for curve_key, curve_value in value.items():
                                                if isinstance(curve_value, np.ndarray):
                                                    curve_subgroup.create_dataset(curve_key, data=curve_value)
                                    elif key in ['class_roc_curves', 'class_pr_curves']:
                                        # Save class-wise curve dicts: class_k -> {curve arrays}
                                        if value:
                                            class_group = round_group.create_group(key)
                                            for class_key, class_curve in value.items():
                                                if isinstance(class_curve, dict):
                                                    cg = class_group.create_group(class_key)
                                                    for curve_key, curve_value in class_curve.items():
                                                        if isinstance(curve_value, np.ndarray):
                                                            cg.create_dataset(curve_key, data=curve_value)
                                    elif key in ['roc_curves', 'pr_curves']:
                                        # Skip list of curves for now (too complex)
                                        continue
                                    else:
                                        # Try to save simple dicts
                                        try:
                                            round_group.create_dataset(key, data=str(value))
                                        except Exception:
                                            pass
                                elif isinstance(value, (list, tuple)):
                                    if key in ('client_roc_curves', 'client_pr_curves'):
                                        try:
                                            curves_group = round_group.create_group(key)
                                            for client_idx, curve in enumerate(value):
                                                if curve is not None and isinstance(curve, dict):
                                                    cg = curves_group.create_group(f'client_{client_idx}')
                                                    for ck, cv in curve.items():
                                                        if isinstance(cv, np.ndarray):
                                                            cg.create_dataset(ck, data=cv)
                                        except Exception:
                                            pass
                                    else:
                                        try:
                                            round_group.create_dataset(key, data=np.array(value))
                                        except Exception:
                                            pass
                            except Exception as e:
                                print(f"Warning: Could not save {key}: {e}")
                
                # Save FL-specific metrics
                fl_metrics = self.fl_metrics_tracker.get_all_fl_metrics()
                if fl_metrics:
                    fl_group = hf.create_group('fl_metrics')
                    
                    # Convergence metrics
                    if 'convergence' in fl_metrics:
                        conv_group = fl_group.create_group('convergence')
                        for key, value in fl_metrics['convergence'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    conv_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    conv_group.create_dataset(key, data=np.string_(value))
                                elif isinstance(value, list):
                                    conv_group.create_dataset(key, data=np.array(value))
                            except Exception:
                                pass
                    
                    # Communication metrics
                    if 'communication' in fl_metrics:
                        comm_group = fl_group.create_group('communication')
                        for key, value in fl_metrics['communication'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    comm_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    comm_group.create_dataset(key, data=np.string_(value))
                                elif isinstance(value, list):
                                    comm_group.create_dataset(key, data=np.array(value))
                            except Exception:
                                pass
                    
                    # Computation time metrics
                    if 'computation' in fl_metrics:
                        comp_group = fl_group.create_group('computation')
                        for key, value in fl_metrics['computation'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    comp_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    comp_group.create_dataset(key, data=np.string_(value))
                                elif isinstance(value, list):
                                    comp_group.create_dataset(key, data=np.array(value))
                            except Exception:
                                pass

                    if 'personalization' in fl_metrics:
                        pers_group = fl_group.create_group('personalization')
                        for key, value in fl_metrics['personalization'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    pers_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    pers_group.create_dataset(key, data=np.string_(value))
                            except Exception:
                                pass

                    if 'model' in fl_metrics:
                        model_group = fl_group.create_group('model')
                        for key, value in fl_metrics['model'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    model_group.create_dataset(key, data=value)
                                elif isinstance(value, str):
                                    model_group.create_dataset(key, data=np.string_(value))
                            except Exception:
                                pass

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def test_metrics_detailed(self):
        """
        Collect detailed metrics from all clients and aggregate them, including per-class and per-client breakdowns.
        
        Returns:
            dict: Aggregated metrics across all clients
        """
        all_detailed_metrics = []
        total_samples = 0
        
        for c in self.clients:
            try:
                metrics_dict = c.test_metrics_detailed()
                all_detailed_metrics.append(metrics_dict)
                total_samples += metrics_dict.get('test_samples', 0)
            except Exception as e:
                print(f"Error getting detailed metrics from client {c.id}: {e}")
                continue
        
        if not all_detailed_metrics:
            return {}

        client_sample_counts = [float(m.get('test_samples', 0)) for m in all_detailed_metrics]
        total_weight = float(np.sum(client_sample_counts))

        def weighted_metric(metric_key):
            vals = [m.get(metric_key, np.nan) for m in all_detailed_metrics]
            num = 0.0
            den = 0.0
            for v, w in zip(vals, client_sample_counts):
                if isinstance(v, (int, float, np.floating)) and not np.isnan(v) and w > 0:
                    num += float(v) * float(w)
                    den += float(w)
            return (num / den) if den > 0 else np.nan
        
        # Aggregate metrics across clients
        aggregated_metrics = {
            'total_test_samples': total_samples,
            'num_clients': len(all_detailed_metrics),
            # Accuracy and micro-accuracy should be sample-weighted, not plain client mean.
            'accuracy': weighted_metric('accuracy'),
            'accuracy_micro': weighted_metric('accuracy_micro'),
            'accuracy_macro': np.nanmean([m.get('accuracy_macro', np.nan) for m in all_detailed_metrics if 'accuracy_macro' in m]),
            'f1_macro': np.nanmean([m['f1_macro'] for m in all_detailed_metrics if 'f1_macro' in m]),
            'f1_micro': np.nanmean([m['f1_micro'] for m in all_detailed_metrics if 'f1_micro' in m]),
            'f1_weighted': np.nanmean([m['f1_weighted'] for m in all_detailed_metrics if 'f1_weighted' in m]),
            'precision_macro': np.nanmean([m['precision_macro'] for m in all_detailed_metrics if 'precision_macro' in m]),
            'precision_micro': np.nanmean([m['precision_micro'] for m in all_detailed_metrics if 'precision_micro' in m]),
            'precision_weighted': np.nanmean([m['precision_weighted'] for m in all_detailed_metrics if 'precision_weighted' in m]),
            'recall_macro': np.nanmean([m['recall_macro'] for m in all_detailed_metrics if 'recall_macro' in m]),
            'recall_micro': np.nanmean([m['recall_micro'] for m in all_detailed_metrics if 'recall_micro' in m]),
            'recall_weighted': np.nanmean([m['recall_weighted'] for m in all_detailed_metrics if 'recall_weighted' in m]),
            'sensitivity_macro': np.nanmean([m.get('sensitivity_macro', np.nan) for m in all_detailed_metrics if 'sensitivity_macro' in m]),
            'sensitivity_micro': np.nanmean([m.get('sensitivity_micro', np.nan) for m in all_detailed_metrics if 'sensitivity_micro' in m]),
            'sensitivity_weighted': np.nanmean([m.get('sensitivity_weighted', np.nan) for m in all_detailed_metrics if 'sensitivity_weighted' in m]),
            'specificity_macro': np.nanmean([m.get('specificity_macro', np.nan) for m in all_detailed_metrics if 'specificity_macro' in m]),
            'specificity_micro': np.nanmean([m.get('specificity_micro', np.nan) for m in all_detailed_metrics if 'specificity_micro' in m]),
            'specificity_weighted': np.nanmean([m.get('specificity_weighted', np.nan) for m in all_detailed_metrics if 'specificity_weighted' in m]),
            'cohen_kappa': np.nanmean([m['cohen_kappa'] for m in all_detailed_metrics if 'cohen_kappa' in m]),
            'matthews_cc': np.nanmean([m['matthews_cc'] for m in all_detailed_metrics if 'matthews_cc' in m]),
            'auc_roc': np.nanmean([m.get('auc_roc', np.nan) for m in all_detailed_metrics if 'auc_roc' in m]),
            'auc_pr': np.nanmean([m.get('auc_pr', np.nan) for m in all_detailed_metrics if 'auc_pr' in m]),
            'brier_score': np.nanmean([m.get('brier_score', np.nan) for m in all_detailed_metrics if 'brier_score' in m]),
        }

        # Store per-client metrics
        client_accuracies = [m['accuracy'] for m in all_detailed_metrics if 'accuracy' in m]
        if client_accuracies:
            aggregated_metrics['client_accuracy_by_client'] = client_accuracies
            aggregated_metrics['client_accuracy_mean'] = float(np.nanmean(client_accuracies))
            aggregated_metrics['client_accuracy_variance'] = float(np.nanvar(client_accuracies))
            
            # Store other per-client metrics (F1, Precision, Recall, AUC, etc.)
            for metric_key in ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'sensitivity_macro', 'specificity_macro', 'auc_roc', 'auc_pr']:
                client_values = [m.get(metric_key, np.nan) for m in all_detailed_metrics]
                valid_values = [v for v in client_values if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_values:
                    aggregated_metrics[f'client_{metric_key}_list'] = client_values
        
        # Add per-class metrics from aggregated confusion matrix
        if any('confusion_matrix' in m for m in all_detailed_metrics):
            try:
                # Aggregate confusion matrices
                cm_total = np.zeros((self.num_classes, self.num_classes))
                for m in all_detailed_metrics:
                    if 'confusion_matrix' in m:
                        cm_total += m['confusion_matrix']
                aggregated_metrics['confusion_matrix'] = cm_total

                # Prefer global metrics derived from the aggregated confusion matrix.
                total_cm = float(np.sum(cm_total))
                if total_cm > 0:
                    accuracy_micro_global = float(np.trace(cm_total) / total_cm)
                    aggregated_metrics['accuracy'] = accuracy_micro_global
                    aggregated_metrics['accuracy_micro'] = accuracy_micro_global

                    row_sums = np.sum(cm_total, axis=1).astype(np.float64)
                    per_class_recall = np.divide(
                        np.diag(cm_total).astype(np.float64),
                        row_sums,
                        out=np.zeros_like(row_sums, dtype=np.float64),
                        where=row_sums > 0
                    )
                    # Balanced accuracy: mean recall over classes present in evaluation.
                    valid = row_sums > 0
                    if np.any(valid):
                        aggregated_metrics['accuracy_macro'] = float(np.mean(per_class_recall[valid]))
                
                # Compute and store per-class metrics from confusion matrix
                aggregated_metrics['per_class_metrics'] = {}
                for class_id in range(self.num_classes):
                    tp = cm_total[class_id, class_id]
                    fp = cm_total[:, class_id].sum() - tp
                    fn = cm_total[class_id, :].sum() - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    support = int(cm_total[class_id, :].sum())
                    
                    aggregated_metrics['per_class_metrics'][class_id] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'support': support
                    }
            except Exception as e:
                print(f"Warning: Could not aggregate confusion matrices: {e}")
        
        # Add ROC and PR curves if available
        # Store both individual curves and a representative curve (first valid one)
        # Also store per-client indexed curves (preserves client ID mapping)
        if any('roc_curve' in m for m in all_detailed_metrics):
            try:
                roc_curves = [m['roc_curve'] for m in all_detailed_metrics if m.get('roc_curve') is not None]
                if roc_curves:
                    # Store the list of all curves (backward-compat, unindexed)
                    aggregated_metrics['roc_curves'] = roc_curves
                    # Also store the first valid curve for direct plotting
                    if isinstance(roc_curves[0], dict) and 'fpr' in roc_curves[0]:
                        aggregated_metrics['roc_curve'] = roc_curves[0]
                # Per-client indexed curves (None for clients without curves)
                aggregated_metrics['client_roc_curves'] = [m.get('roc_curve') for m in all_detailed_metrics]
            except Exception:
                pass
        
        if any('pr_curve' in m for m in all_detailed_metrics):
            try:
                pr_curves = [m['pr_curve'] for m in all_detailed_metrics if m.get('pr_curve') is not None]
                if pr_curves:
                    # Store the list of all curves
                    aggregated_metrics['pr_curves'] = pr_curves
                    # Also store the first valid curve for direct plotting
                    if isinstance(pr_curves[0], dict) and 'precision' in pr_curves[0]:
                        aggregated_metrics['pr_curve'] = pr_curves[0]
                # Per-client indexed curves (None for clients without curves)
                aggregated_metrics['client_pr_curves'] = [m.get('pr_curve') for m in all_detailed_metrics]
            except Exception:
                pass

        # Add class-wise ROC/PR curves and class-wise AUC (averaged over clients)
        try:
            class_roc_curves = {}
            class_pr_curves = {}
            class_auc_roc_by_class = []
            class_auc_pr_by_class = []

            for class_id in range(self.num_classes):
                class_key = f'class_{class_id}'

                # Use first available per-class curve as representative for visualization.
                for m in all_detailed_metrics:
                    class_roc = m.get('class_roc_curves', {})
                    if isinstance(class_roc, dict) and class_key in class_roc:
                        class_roc_curves[class_key] = class_roc[class_key]
                        break

                for m in all_detailed_metrics:
                    class_pr = m.get('class_pr_curves', {})
                    if isinstance(class_pr, dict) and class_key in class_pr:
                        class_pr_curves[class_key] = class_pr[class_key]
                        break

                roc_vals = []
                pr_vals = []
                for m in all_detailed_metrics:
                    auc_roc_list = m.get('class_auc_roc_by_class', [])
                    auc_pr_list = m.get('class_auc_pr_by_class', [])
                    if isinstance(auc_roc_list, (list, tuple, np.ndarray)) and class_id < len(auc_roc_list):
                        v = auc_roc_list[class_id]
                        if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                            roc_vals.append(float(v))
                    if isinstance(auc_pr_list, (list, tuple, np.ndarray)) and class_id < len(auc_pr_list):
                        v = auc_pr_list[class_id]
                        if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                            pr_vals.append(float(v))

                class_auc_roc_by_class.append(float(np.mean(roc_vals)) if roc_vals else np.nan)
                class_auc_pr_by_class.append(float(np.mean(pr_vals)) if pr_vals else np.nan)

            if class_roc_curves:
                aggregated_metrics['class_roc_curves'] = class_roc_curves
            if class_pr_curves:
                aggregated_metrics['class_pr_curves'] = class_pr_curves
            aggregated_metrics['class_auc_roc_by_class'] = class_auc_roc_by_class
            aggregated_metrics['class_auc_pr_by_class'] = class_auc_pr_by_class
        except Exception:
            pass
        
        return aggregated_metrics

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        
        # Collect detailed metrics
        try:
            detailed_metrics = self.test_metrics_detailed()
            if detailed_metrics:
                self.rs_detailed_metrics.append(detailed_metrics)
                # Print detailed metrics
                print("\n--- Detailed Metrics ---")
                print(f"F1-Score (Macro): {detailed_metrics.get('f1_macro', 0):.4f}")
                print(f"F1-Score (Weighted): {detailed_metrics.get('f1_weighted', 0):.4f}")
                print(f"Precision (Macro): {detailed_metrics.get('precision_macro', 0):.4f}")
                print(f"Recall (Macro): {detailed_metrics.get('recall_macro', 0):.4f}")
                if 'sensitivity_macro' in detailed_metrics:
                    print(f"Sensitivity (Macro): {detailed_metrics.get('sensitivity_macro', 0):.4f}")
                if 'specificity_macro' in detailed_metrics:
                    print(f"Specificity (Macro): {detailed_metrics.get('specificity_macro', 0):.4f}")
                print(f"Kappa Score: {detailed_metrics.get('cohen_kappa', 0):.4f}")
                print(f"Matthews CC: {detailed_metrics.get('matthews_cc', 0):.4f}")
                print(f"AUC-ROC: {detailed_metrics.get('auc_roc', 0):.4f}")
                print(f"AUC-PR: {detailed_metrics.get('auc_pr', 0):.4f}")
                _brier = detailed_metrics.get('brier_score', float('nan'))
                try:
                    if not np.isnan(float(_brier)):
                        print(f"Brier Score: {float(_brier):.4f}")
                except (TypeError, ValueError):
                    pass
                if 'client_accuracy_variance' in detailed_metrics:
                    print(f"Client Accuracy Variance: {detailed_metrics.get('client_accuracy_variance', 0):.6f}")
        except Exception as e:
            print(f"Error collecting detailed metrics: {e}")
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        accs = []
        f1s = []
        aucs = []
        recalls = []
        for c in self.new_clients:
            #ct, ns, auc = c.test_metrics()
            acc, f1, auc, recall = c.test_metrics()
            accs.append(acc)
            f1s.append(f1)
            aucs.append(auc)
            recalls.append(recall)
            # tot_correct.append(ct*1.0)
            # tot_auc.append(auc)
            # num_samples.append(ns)
            print(f"Mean Accuracy: {np.mean(accs):.4f}")
            print(f"Mean Macro-F1: {np.mean(f1s):.4f}")
            print(f"Mean AUC: {np.mean(aucs):.4f}")

            mean_recall = np.mean(np.stack(recalls), axis=0)
            print("Per-class Recall:", mean_recall)


        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc