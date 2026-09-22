#!/usr/bin/env python
"""
Baseline training script (OptGNN/LiftMP).

Trains the LiftMP baseline network (Yau et al.) as re-implemented inside this
package in model/baselines.py, on top of this package's own data pipeline
(data/loader.py) and problem definitions (problem/problems.py).

This model does NOT use dual (lambda/mu) variables: the loss is a fixed
quadratic penalty on constraint violations, and only gradient information is
backpropagated.  Keep the CLI interface unchanged.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import construct_loaders
from problem.problems import get_problem
from model.baselines import build_liftmp_baseline, get_liftmp_loss_fn
from utils.parsing import parse_train_args
from model.saving import save_model
from problem.baselines import random_hyperplane_projector
from torch_geometric.transforms import AddRandomWalkPE


def featurize_batch(args, batch):
    """Featurize a batch of graphs for LiftMP."""
    batch = batch.to(args.device)

    N = batch.num_nodes

    # build x
    if args.positional_encoding is None or args.pe_dimension == 0:
        # generate random vector input
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


def construct_model(args):
    """Construct the LiftMP baseline (no lambda_param needed)."""
    model = build_liftmp_baseline(args)
    optimizer = Adam([{"params": model.parameters(), "lr": args.lr}])
    return model, optimizer


def validate(args, model, val_loader, problem, loss_fn):
    """Validate the model (no lambda_param for LiftMP)."""
    total_loss = 0.
    total_score = 0.
    total_constraint = 0.
    total_count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch.penalty = args.penalty
            if len(batch) == 1:
                datalist = [batch]
            else:
                datalist = batch.to_data_list()

            x_in, batch = featurize_batch(args, batch)

            # LiftMP: no dual variables, fixed quadratic penalty only
            x_out = model(x_in, batch)
            loss = loss_fn(x_out, batch)

            total_loss += float(loss)

            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score)
            x_proj = torch.where(x_proj == 0, 1, x_proj)

            num_zeros = (x_proj == 0).count_nonzero()
            assert num_zeros == 0

            score = problem.score(args, x_proj, batch)
            total_score += float(score)
            total_constraint += float(problem.constraint(x_proj, batch))
            total_count += len(batch)

    return total_loss / total_count, total_score / total_count, total_constraint / total_count


def train(args, model, train_loader, optimizer, loss_fn, val_loader=None, test_loader=None, problem=None):
    """Main training loop for LiftMP (no dual updates)."""
    epochs = args.epochs
    model_folder = args.log_dir

    train_losses = []
    valid_losses = []
    valid_scores = []
    valid_constraints = []
    test_losses = []
    test_scores = []
    test_constraints = []

    model.to(args.device)

    ep = 0
    steps = 0

    while args.stepwise or ep <= epochs:
        start_time = time.time()
        epoch_total_loss = 0.
        epoch_count = 0

        for batch in train_loader:
            args.penalty = args.penalty_s + ep/epochs * (args.penalty_e - args.penalty_s)
            batch.penalty = args.penalty

            # run the model
            x_in, batch = featurize_batch(args, batch)

            # LiftMP: no dual variables, fixed quadratic penalty only
            x_out = model(x_in, batch)

            # get loss
            loss = loss_fn(x_out, batch)

            # run gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate and store average loss for batch
            avg_loss = float(loss) / batch.num_graphs
            train_losses.append(avg_loss)

            # increment epoch loss counters
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
                    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, loss_fn)
                    if len(valid_scores) == 0 or valid_score > max(valid_scores):
                        save_model(model, f"{model_folder}/best_model.pt")
                    valid_losses.append(valid_loss)
                    valid_scores.append(valid_score)
                    valid_constraints.append(valid_constraint)
                    valid_time = time.time() - valid_start_time

                    if test_loader is not None:
                        test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, loss_fn)
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

        if args.stepwise and steps >= args.steps:
            break

        # print average loss for epoch
        epoch_time = time.time() - start_time
        epoch_avg_loss = epoch_total_loss / epoch_count
        print(f"epoch={ep} t={epoch_time:0.2f} steps={steps} epoch_avg_loss={epoch_avg_loss:0.2f}")

        if not args.stepwise:
            if (args.valid_freq != 0 and ep % args.valid_freq == 0):
                valid_start_time = time.time()
                valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, loss_fn)
                valid_losses.append(valid_loss)
                valid_scores.append(valid_score)
                valid_constraints.append(valid_constraint)
                valid_time = time.time() - valid_start_time

                if test_loader is not None:
                    test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, loss_fn)
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
    start_time = time.perf_counter()
    args = parse_train_args()
    print(args)
    torch.manual_seed(args.seed)
    # replace ':' with '_' in the param hash for cross-platform path safety
    args.log_dir = args.log_dir.replace('paramhash:', 'paramhash_')
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))

    # get data, model
    train_loader, val_loader, test_loader, _mu_unused = construct_loaders(args)
    model, optimizer = construct_model(args)
    loss_fn = get_liftmp_loss_fn(args)
    problem = get_problem(args)  # score / constraint for validation metrics

    # train model
    train(args, model, train_loader, optimizer, loss_fn,
          val_loader=val_loader, test_loader=test_loader, problem=problem)

    # run final validation
    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, loss_fn)

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
