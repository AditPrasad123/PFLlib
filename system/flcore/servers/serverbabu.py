import time
import random
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import torch

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


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n---Round number: {i}---")
                print("Evaluate global model")
                self.evaluate()
            print("Client done.")
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

            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            print("Round done.")

        print("\nFinal Results:")
        print(f"Best Accuracy: {max(self.rs_test_acc):.4f}")
        print(f"Final Accuracy: {self.rs_test_acc[-1]:.4f}")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        for client in self.clients:
            client.fine_tune()
        
        torch.save(self.clients, "clients_finetune.pt")
        print("Saved clients after fine-tuning.")
        for client in self.clients:
            print(f"Client {client.id} head weights sum:",
            sum(p.sum().item() for p in client.model.head.parameters()))

        print("\nFinal Personalized Results (FedBABU) [Baseline]:")
        self.evaluate()

        # 🔥 ADD THIS (TTFT)
        for client in self.clients:
            client.test_time_finetune()
        torch.save(self.clients, "clients_ttft.pt")
        print("Saved clients after TTFT.")
        print("\nFinal Personalized Results (FedBABU + TTFT):")
        self.evaluate()
  
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientBABU)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

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