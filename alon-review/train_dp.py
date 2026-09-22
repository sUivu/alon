#!/usr/bin/env python
"""
PDGNO training script (ALMGNN).
Copied from pdgno and adapted for PDGNO experiments.

This script trains the ALMGNN model which DOES use mu_param (lambda) terms.
PDGNO uses augmented Lagrangian method with dual variable updates.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from data.loader import construct_loaders
from problem.problems import get_problem
from model.models import DecoupledAutogradLayer, ALMGNN_Network
from utils.parsing import parse_train_args
from model.saving import save_model
from problem.baselines import random_hyperplane_projector
from torch_geometric.transforms import AddRandomWalkPE


def featurize_batch(args, batch):
    """Featurize a batch of graphs for ALMGNN."""
    batch = batch.to(args.device)

    N = batch.num_nodes
    num_edges = batch.edge_index.shape[1]

    # build x
    if args.positional_encoding is None or args.pe_dimension == 0:
        x_in = torch.randn((N, args.rank), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
    elif args.positional_encoding == 'laplacian_eigenvector':
        x_in = torch.randn((N, args.rank - args.pe_dimension), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.laplacian_eigenvector_pe.to(args.device)[:, :args.pe_dimension]
        sign = -1 + 2 * torch.randint(0, 2, (args.pe_dimension, ), device=args.device)
        pe *= sign
        x_in = torch.cat((x_in, pe), 1)
    elif args.positional_encoding == 'random_walk':
        if not hasattr(batch, 'random_walk_pe'):
            batch = AddRandomWalkPE(walk_length=args.pe_dimension)(batch.to(args.device))
        x_in = torch.randn((N, args.rank - args.pe_dimension), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.random_walk_pe.to(args.device)[:, :args.pe_dimension]
        x_in = torch.cat((x_in, pe), 1)
    else:
        raise ValueError(f"Invalid transform passed into featurize_batch: {args.transform}")

    return x_in, batch


def construct_model(args, mu_param):
    """Construct ALMGNN model (needs mu_param for augmented Lagrangian)."""
    grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
    model = ALMGNN_Network(
        grad_layer=grad_layer,
        in_channels=args.rank,
        num_layers=args.num_layers,
        lift_ratio=args.lift_ratio,
    )
    optimizer = Adam([{"params": model.parameters(), "lr": args.lr}])
    return model, optimizer


def validate(args, model, val_loader, problem, mu_param=None):
    """Validate the ALMGNN model (with mu_param)."""
    total_loss = 0.
    total_score = 0.
    total_constraint = 0.
    total_count = 0

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch.penalty = args.penalty
            if len(batch) == 1:
                datalist = [batch]
            else:
                datalist = batch.to_data_list()

            x_in, batch = featurize_batch(args, batch)
            
            # ALMGNN: 使用 mu_param (lambda项)
            x_out = model(x_in, batch, mu_param)
            loss = problem.loss(x_out, batch, mu_param, mode="valid")

            total_loss += float(loss)

            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score)
            x_proj = torch.where(x_proj == 0, 1, x_proj)

            num_zeros = (x_proj == 0).count_nonzero()
            assert num_zeros == 0

            score = problem.score(args, x_proj, batch)
            total_score += float(score)
            total_constraint += float(problem.constraint(x_proj, batch))
            total_count += len(batch)

    model.train()
    return total_loss / total_count, total_score / total_count, total_constraint / total_count


def train(args, model, train_loader, optimizer, problem, val_loader=None, test_loader=None, mu_param=None):
    """
    Main training loop for ALMGNN (with mu_param updates).
    
    Unlike LiftMP, ALMGNN uses augmented Lagrangian method:
    - Loss includes penalty term: L(x, λ) = f(x) + λ * constraint(x) + (ρ/2) * constraint(x)^2
    - mu_param (λ) is updated periodically based on constraint violations
    """
    epochs = args.epochs
    model_folder = args.log_dir

    train_losses = []
    valid_losses = []
    valid_scores = []
    valid_constraints = []
    test_losses = []
    test_scores = []
    test_constraints = []
    lambda_history = []
    lambda_history.append(mu_param.detach().cpu().clone())

    model.to(args.device)

    ep = 0
    steps = 0

    while args.stepwise or ep <= epochs:
        start_time = time.time()
        epoch_total_loss = 0.
        epoch_count = 0
        add_mu_param = torch.zeros_like(mu_param)

        for batch in train_loader:
            # Linear annealing of penalty parameter
            if args.linear_annealing:
                args.penalty = args.penalty_s + ep/epochs * (args.penalty_e - args.penalty_s)
            batch.penalty = args.penalty

            # run the model
            x_in, batch = featurize_batch(args, batch)
            
            # ALMGNN: 使用 mu_param 计算增强拉格朗日损失
            x_out = model(x_in, batch, mu_param)

            # get loss with penalty term
            loss, constraint_val = problem.loss(x_out, batch, mu_param, return_constraint=True)
            
            # Accumulate constraint violations for mu_param update
            add_mu_param[batch.node_global_idx] += constraint_val.detach().clone()

            # run gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = float(loss) / batch.num_graphs
            train_losses.append(avg_loss)
            epoch_total_loss += float(loss)
            epoch_count += batch.num_graphs
            steps += 1

            if args.stepwise:
                if args.infinite and steps % 100 == 0:
                    epoch_time = time.time() - start_time
                    epoch_avg_loss = epoch_total_loss / epoch_count
                    print(f"steps={steps} t={epoch_time:0.2f} epoch_avg_loss={epoch_avg_loss:0.2f}")
                    start_time = time.time()
                    epoch_total_loss = 0.
                    epoch_count = 0

                if args.valid_freq != 0 and steps % args.valid_freq == 0:
                    valid_start_time = time.time()
                    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, mu_param)
                    if len(valid_scores) == 0 or valid_score > max(valid_scores):
                        save_model(model, f"{model_folder}/best_model.pt")
                    valid_losses.append(valid_loss)
                    valid_scores.append(valid_score)
                    valid_constraints.append(valid_constraint)
                    valid_time = time.time() - valid_start_time
                    
                    if test_loader is not None:
                        test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, mu_param)
                        test_losses.append(test_loss)
                        test_scores.append(test_score)
                        test_constraints.append(test_constraint)
                    else:
                        test_loss = np.inf
                        test_score = -np.inf

                    print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f}\n"
                          f"  valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}\n"
                          f"  test_loss={test_loss} test_score={test_score} test_constraint={test_constraint}")

                if steps >= args.steps:
                    break

        # Update mu_param (lambda) periodically
        # This is the key difference from LiftMP - ALMGNN updates dual variables
        if not ep % args.lambda_freq:
            with torch.no_grad():
                print("ep:", ep, 
                      "mu_param_norm:", torch.norm(mu_param),
                      "updating_mu_param_norm:", torch.norm(add_mu_param),
                      "max:", add_mu_param.max().item(),
                      "min:", add_mu_param.min().item())
                add_mu_param.clamp_(min=0.0)
                mu_param += args.lambda_ratio * add_mu_param
                lambda_history.append(mu_param.detach().clone())

        if args.stepwise and steps >= args.steps:
            break

        epoch_time = time.time() - start_time
        epoch_avg_loss = epoch_total_loss / epoch_count
        print(f"epoch={ep} t={epoch_time:0.2f} steps={steps} epoch_avg_loss={epoch_avg_loss:0.2f}")

        if not args.stepwise:
            if (args.valid_freq != 0 and ep % args.valid_freq == 0):
                valid_start_time = time.time()
                valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, mu_param)
                valid_losses.append(valid_loss)
                valid_scores.append(valid_score)
                valid_constraints.append(valid_constraint)
                valid_time = time.time() - valid_start_time
                
                if test_loader is not None:
                    test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, mu_param)
                    test_losses.append(test_loss)
                    test_scores.append(test_score)
                    test_constraints.append(test_constraint)
                else:
                    test_loss = np.inf
                    test_score = -np.inf
                    
                print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f}\n"
                      f"  valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}\n"
                      f"  test_loss={test_loss} test_score={test_score} test_constraint={test_constraint}")

        ep += 1

    # Save final model
    save_model(model, f"{model_folder}/final_model.pt")
    return model


if __name__ == '__main__':
    # parse args
    start_time = time.perf_counter()
    args = parse_train_args()
    print(args)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Convert to Windows-compatible path (replace : with _ in hash)
    args.log_dir = args.log_dir.replace('paramhash:', 'paramhash_')
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))

    # get data, model (includes mu_param initialization)
    train_loader, val_loader, test_loader, mu_param = construct_loaders(args)
    model, optimizer = construct_model(args, mu_param)
    problem = get_problem(args)

    # train model (with mu_param updates)
    train(args, model, train_loader, optimizer, problem, 
          val_loader=val_loader, test_loader=test_loader, mu_param=mu_param)

    # run final validation
    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, mu_param)

    # write "done" file
    with open(os.path.join(args.log_dir, 'done.txt'), 'w') as file:
        file.write(f"{valid_loss}\n")
        file.write(f"valid_score: {valid_score}\n")
        file.write(f"valid_constraint: {valid_constraint}\n")
        file.write("done.\n")
    
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    print(f"Total runtime: {hours}h {minutes}m {seconds:.2f}s")
