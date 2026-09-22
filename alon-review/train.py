import json
import os

import torch

from data.loader import construct_loaders
from problem.problems import get_problem
from model.models import construct_model
from model.saving import save_model
from utils.parsing import parse_train_args
from model.training import train, validate
import time
import numpy as np
if __name__ == '__main__':
    # parse args
    # record start time (high-precision timer)
    start_time = time.perf_counter()
    args = parse_train_args()
    print(args)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))

    # get data, model
    train_loader, val_loader, test_loader, mu_param = construct_loaders(args)
    model, optimizer = construct_model(args, mu_param)
    problem = get_problem(args)
    
    # train model
    train(args, model, train_loader, optimizer, problem, val_loader=val_loader, test_loader=test_loader, mu_param=mu_param)

    # save final model checkpoint for OOD evaluation
    save_model(model, f"{args.log_dir}/final_model.pt")
    print(f"Model saved to {args.log_dir}/final_model.pt")

    # run final validation
    valid_loss = validate(args, model, val_loader, problem, mu_param=mu_param)

    # write "done" file
    with open(os.path.join(args.log_dir, 'done.txt'), 'w') as file:
        file.write(f"{valid_loss}\n")
        file.write("done.\n")
    
    end_time = time.perf_counter()

    # compute total elapsed time (seconds)
    elapsed_seconds = end_time - start_time

    # convert to hours, minutes, seconds
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60

    # print result (two decimal places, accurate to hundredths of a second)
    print(f"Total program runtime: {hours}h {minutes}m {seconds:.2f}s")
    # 3. get problem type (e.g. VertexCoverProblem)
    
    
    # problem_cls = get_problem(args)
    
    # print(f"Starting Gurobi solve on dataset: {args.dataset}, problem: {args.problem_type}")
    # print(f"Test set size: {len(test_loader.dataset)}")

    # results = []
    # runtimes = []

    # # 4. solve graph by graph
    # # Gurobi can only process one graph at a time, so batch_size should be 1
    # # or batches should be split manually
    # for i, batch in enumerate(test_loader):
    #     # split the Batch into individual graph objects
    #     examples = batch.to_data_list()
    #     for j, example in enumerate(examples):
    #         start_t = time.time()

    #         # call the gurobi interface in your problem class
    #         # returns: x_vals (solution vector), status (solver status), runtime (solve time)
    #         x_vals, status, runtime = problem_cls.gurobi(args, example)

    #         # compute the Score of this solution (for MVC: -number of selected vertices)
    #         # note: Gurobi returns np.array; convert to torch before passing to score
    #         score = problem_cls.score(args, torch.from_numpy(x_vals).float(), example)

    #         results.append(score)
    #         runtimes.append(runtime)

    #         if (len(results)) % 10 == 0:
    #             print(f"Done: {len(results)} graphs, current mean Score: {np.mean(results):.4f}")
    # # 5. print final statistics
    # print("\n" + "="*30)
    # print(f"Gurobi final statistics ({args.problem_type}):")
    # print(f"Mean optimum (Score): {np.mean(results):.4f}")
    # print(f"Std dev: {np.std(results):.4f}")
    # print(f"Mean solve time: {np.mean(runtimes):.4f}s")
    # print("="*30)