import time
from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from threading import Thread


class FedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProx)


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
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
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
            self.fl_metrics_tracker.add_round_time(round_time)
            self.fl_metrics_tracker.add_local_computation_time(round_time)

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

        print("\n-------------Evaluate personalized + TTFT models-------------")
        self.evaluate()

        if baseline_acc is not None and len(self.rs_test_acc) > 0:
            personalized_acc = self.rs_test_acc[-1]
            self.fl_metrics_tracker.set_personalization_metrics(
                baseline_acc, personalized_acc
            )

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
