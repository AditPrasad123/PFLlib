import time
import random
import os
import torch
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from threading import Thread


class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBABU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def save_personalized_client_models(self):
        """Save full personalized client models (base + head) for explainability analysis."""
        base_dir = os.path.join("models", self.dataset, "FedBABU_personalized_clients", f"run_{self.times}")
        os.makedirs(base_dir, exist_ok=True)

        for client in self.clients:
            file_path = os.path.join(base_dir, f"client_{client.id}.pt")
            torch.save(client.model, file_path)

        print(f"Saved personalized client models to: {base_dir}")


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                # Track accuracy for convergence analysis
                if len(self.rs_test_acc) > 0:
                    self.fl_metrics_tracker.add_test_accuracy(self.rs_test_acc[-1])

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            round_time = time.time() - s_t
            self.Budget.append(round_time)
            # Track time metrics
            self.fl_metrics_tracker.add_round_time(round_time)
            self.fl_metrics_tracker.add_local_computation_time(round_time)
            
            # Calculate and track communication cost
            communication_cost = self.calculate_communication_cost()
            self.fl_metrics_tracker.add_communication_cost(communication_cost)
            
            print('-'*25, 'time cost', '-'*25, round_time)
            print('-'*25, f'communication cost: {communication_cost / (1024*1024):.2f} MB', '-'*25)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        baseline_acc = self.rs_test_acc[-1] if len(self.rs_test_acc) > 0 else None

        for client in self.clients:
            client.fine_tune()
        print("\n--- Test-time fine-tuning ---")
        for client in self.clients:
            client.test_time_finetune()

        #print("\n-------------Evaluate fine-tuned personalized models-------------")
        print("\n-------------Evaluate personalized + TTFT models-------------")

        self.evaluate()

        if baseline_acc is not None and len(self.rs_test_acc) > 0:
            personalized_acc = self.rs_test_acc[-1]
            self.fl_metrics_tracker.set_personalization_metrics(
                baseline_acc, personalized_acc
            )

        self.save_personalized_client_models()
        
        # Print FL metrics summary
        print("\n" + "="*50)
        print("FEDERATED LEARNING METRICS SUMMARY")
        print("="*50)
        fl_metrics = self.fl_metrics_tracker.get_all_fl_metrics()
        
        if fl_metrics['convergence']:
            print("\nConvergence Metrics:")
            print(f"  Final Accuracy: {fl_metrics['convergence']['final_accuracy']:.4f}")
            print(f"  Initial Accuracy: {fl_metrics['convergence']['initial_accuracy']:.4f}")
            print(f"  Improvement: {fl_metrics['convergence']['improvement']:.4f}")
            print(f"  Rounds to 95% Convergence: {fl_metrics['convergence']['rounds_to_convergence']}")
        
        if fl_metrics['computation']:
            print("\nComputation Time (per Round):")
            print(f"  Average: {fl_metrics['computation']['avg_time_per_round']:.4f}s")
            print(f"  Min: {fl_metrics['computation']['min_time_per_round']:.4f}s")
            print(f"  Max: {fl_metrics['computation']['max_time_per_round']:.4f}s")
            print(f"  Total: {fl_metrics['computation']['total_time_minutes']:.2f} minutes")
        
        if fl_metrics['communication']:
            print("\nCommunication Overhead (per Round):")
            print(f"  Average: {fl_metrics['communication']['avg_communication_per_round_mb']:.2f} MB")
            print(f"  Total: {fl_metrics['communication']['total_communication_mb']:.2f} MB")

        if 'personalization' in fl_metrics:
            print("\nPersonalization Gain:")
            print(f"  Baseline Accuracy: {fl_metrics['personalization']['baseline_accuracy']:.4f}")
            print(f"  Personalized Accuracy: {fl_metrics['personalization']['personalized_accuracy']:.4f}")
            print(f"  Gain: {fl_metrics['personalization']['personalization_gain']:.4f}")

        if 'model' in fl_metrics:
            print("\nQuantum Model Info:")
            if 'n_qubits' in fl_metrics['model']:
                print(f"  Qubits: {fl_metrics['model']['n_qubits']}")
            if 'circuit_depth' in fl_metrics['model']:
                print(f"  Circuit Depth: {fl_metrics['model']['circuit_depth']}")
        
        print("="*50 + "\n")
        
        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientBABU)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()
            
    

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
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples