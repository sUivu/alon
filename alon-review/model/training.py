import numpy as np
import os
import time

import torch
import torch.nn.functional as F

from model.saving import save_model
from problem.baselines import random_hyperplane_projector

from torch_geometric.transforms import AddRandomWalkPE
import time
import numpy as np
def featurize_batch(args, batch):
    batch = batch.to(args.device)

    N = batch.num_nodes
    num_edges = batch.edge_index.shape[1]

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
        # XXX add the random walk PE here
        if not hasattr(batch, 'random_walk_pe'):
            batch = AddRandomWalkPE(walk_length=args.pe_dimension)(batch.to(args.device))
        x_in = torch.randn((N, args.rank - args.pe_dimension), dtype=torch.float, device=args.device)
        x_in = F.normalize(x_in, dim=1)
        pe = batch.random_walk_pe.to(args.device)[:, :args.pe_dimension]
        x_in = torch.cat((x_in, pe), 1)
    else:
        raise ValueError(f"Invalid transform passed into featurize_batch: {args.transform}")

    # TODO handling multi-penalty situations -- shouldn't be in featurize
    # batch.penalty = args.penalty

    return x_in, batch

# measure and return the validation loss
def validate(args, model, val_loader, problem, mu_param=None):
    total_loss = 0.
    total_score = 0.
    total_constraint = 0.
    total_count = 0
    opt_ = 0.
    opt__ = 0.
    gap = 0.
    eigen = 0.
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch.penalty = args.penalty
            if len(batch) == 1:
                datalist = [batch]
            else:
                datalist = batch.to_data_list()

            x_in, batch = featurize_batch(args, batch)
            
            if args.model_type in ['GIN', 'GAT', 'GCNN', 'GatedGCNN']:
                x_out = model(x_in, batch)
            else:   
                x_out = model(x_in, batch, mu_param)
            loss = problem.loss(x_out, batch, mu_param, mode="valid")
            
            total_loss += float(loss)

            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score)

            # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
            x_proj = torch.where(x_proj == 0, 1, x_proj)

            num_zeros = (x_proj == 0).count_nonzero()
            assert num_zeros == 0

            # count the score
            score = problem.score(args, x_proj, batch)
            total_score += float(score)
            total_constraint += float(problem.constraint(x_proj, batch))
            

            with torch.enable_grad():
                x_in_grad = x_in.detach().requires_grad_(True)
                if args.model_type in ['GIN', 'GAT', 'GCNN', 'GatedGCNN']:
                    x_out_grad = model(x_in_grad, batch)
                else:
                    x_out_grad = model(x_in_grad, batch, mu_param)
                loss_grad = problem.loss(x_out_grad, batch, mu_param, mode="valid")
                
                param_snapshot = [p.clone().detach() for p in model.parameters()]
                # Correctly compute the gradient of the loss w.r.t. the input x_in
                grad_x = torch.autograd.grad(loss_grad, x_in_grad, create_graph=False)[0]  # same shape as x_in

                for p, p_snap in zip(model.parameters(), param_snapshot):
                    assert torch.allclose(p, p_snap), "Model parameters were modified!"
                # Reshape grad_x to a 2D matrix for SVD (if original dim > 2)
                if grad_x.dim() > 2:
                    grad_x_2d = grad_x.view(grad_x.size(0), -1)
                else:
                    grad_x_2d = grad_x
                U, S, V = torch.svd(grad_x_2d)          # singular values in descending order
                min_sv = S[-1].item()                    # smallest singular value

                # Compute the scalar inner product (element-wise product of grad_x and x_in_grad, summed)
                inner_product = (grad_x * x_in_grad).sum().item()

                # Update accumulators (all terms are scalars)
                add_opt_ = loss_grad.item() - inner_product + min_sv * len(batch)
                opt_ += add_opt_
                add_gap = - inner_product + min_sv * len(batch)
                gap  += add_gap
                add_opt__ = - score.item() - inner_product + min_sv * len(batch)
                opt__ += add_opt__   # note the use of total_score
                eigen += min_sv
                # print("add_opt_:", add_opt_,"add_opt__:", add_opt__, "add_gap:", add_gap, "eigen:", eigen)   # note the use of total_score
            total_count += len(batch)
            
        # print("opt_:",opt_/total_count)
        # print("opt__:",opt__/total_count)
        # print("gap:",gap/total_count)
        # print("eigen:",eigen/total_count)
    return total_loss / total_count, total_score / total_count, total_constraint / total_count

def train(args, model, train_loader, optimizer, problem, val_loader=None, test_loader=None,mu_param=None):
    '''Main training loop:

    Trains a model with an optimizer for a number of epochs
    '''
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
    lambda_history.append(
        mu_param.detach().cpu().clone()
        )
    model.to(args.device)

    ep = 0
    steps = 0

    while args.stepwise or ep <= epochs:
        start_time = time.time()
        # if ep >= 90:
        #     print("ep:", ep,110)
        # reset epoch average loss counters
        epoch_total_loss = 0.
        epoch_count = 0
        # start_time = time.time()
        add_mu_param = torch.zeros_like(mu_param)
        # if ep >= 90:
        #     print("ep:", ep,111)
        for batch in train_loader:
            # batch.penalty = args.penalty_s + ep/epochs * (args.penalty_e - args.penalty_s)
            if args.linear_annealing:
                args.penalty = args.penalty_s + ep/epochs * (args.penalty_e - args.penalty_s)
            # args.penalty = args.penalty_s + np.cos(ep/epochs * np.pi/2) * (args.penalty_e - args.penalty_s)
            batch.penalty = args.penalty
            # run the model
            x_in, batch = featurize_batch(args, batch)
            # print("steps:", steps)
            if args.model_type in ['GIN', 'GAT', 'GCNN', 'GatedGCNN']:
                x_out = model(x_in, batch)
            else:   
                x_out = model(x_in, batch, mu_param)

            # get loss
            loss, constraint_val = problem.loss(x_out, batch, mu_param, return_constraint=True)
            # if not ep % args.lambda_freq:
            add_mu_param[batch.node_global_idx] += constraint_val.detach().clone()
            # run gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if ep >= 90:
            #     print("ep:", ep, steps, 112)
            # calculate and store average loss for batch
            avg_loss = float(loss) / batch.num_graphs
            train_losses.append(avg_loss)

            # increment epoch loss counters
            epoch_total_loss += float(loss)
            epoch_count += batch.num_graphs

            steps += 1
            # if ep >= 90:
            #     print("ep:", ep, steps)
            if args.stepwise:
                # occasionally print training loss for infinite datasets
                if args.infinite and steps % 100 == 0:
                    epoch_time = time.time() - start_time
                    epoch_avg_loss = epoch_total_loss / epoch_count
                    print(f"steps={steps} t={epoch_time:0.2f} epoch_avg_loss={epoch_avg_loss:0.2f}")

                    start_time = time.time()
                    epoch_total_loss = 0.
                    epoch_count = 0

                # occasionally run validation
                if args.valid_freq != 0 and steps % args.valid_freq == 0:
                    valid_start_time = time.time()
                    model.eval()
                    valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, mu_param)
                    model.train()
                    # save model if it's the current best
                    if len(valid_scores)==0 or valid_score > max(valid_scores):
                        save_model(model, f"{model_folder}/best_model.pt")
                    valid_losses.append(valid_loss)
                    valid_scores.append(valid_score)
                    valid_constraints.append(valid_constraint)
                    valid_time = time.time() - valid_start_time
                    
                    # test
                    if test_loader is not None:
                        model.eval()
                        test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, mu_param)
                        model.train()
                        test_losses.append(test_loss)
                        test_scores.append(test_score)
                        test_constraints.append(test_constraint)
                    else:
                        test_loss = np.inf
                        test_score = -np.inf

                    print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f}\n\
                                valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}\n\
                                test_loss={test_loss} test_score={test_score} test_constraint={test_constraint}")

                # check if training is done
                if steps >= args.steps:
                    break

                # occasionally save model
                # if args.save_freq != 0 and steps % args.save_freq == 0:
                #     save_model(model, f"{model_folder}/model_step{steps}.pt")
        if not ep % args.lambda_freq and ep >= getattr(args, 'dual_warmup', 0):
            with torch.no_grad():
                print("ep:", ep,
                      "mu_param:",torch.norm(mu_param),
                      "updating mu_param_norm:", torch.norm(add_mu_param),
                      "max:", add_mu_param.max().item(),
                      "min:", add_mu_param.min().item(),
                      )
                add_mu_param.clamp_(min=0.0)
                mu_param += args.lambda_ratio * add_mu_param
                if getattr(args, 'mu_cap', None) is not None:
                    mu_param.clamp_(max=args.mu_cap)
                lambda_history.append(mu_param.detach().clone())
            # lambda_param.clamp_(min=0.0)
                

        if args.stepwise and steps >= args.steps:
            break

        # print average loss for epoch
        epoch_time = time.time() - start_time
        epoch_avg_loss = epoch_total_loss / epoch_count
        # epoch_time = time.time() - start_time
        # m, s = divmod(int(epoch_time), 60)
        # print(f"epoch={ep} | time={m}m {s}s | steps={steps} | epoch_avg_loss={epoch_avg_loss:.4f}")
        print(f"epoch={ep} t={epoch_time:0.2f} steps={steps} epoch_avg_loss={epoch_avg_loss:0.2f}")

        if not args.stepwise:
            # occasionally run validation
            if (args.valid_freq != 0 and ep % args.valid_freq == 0):
                valid_start_time = time.time()
                model.eval()
                valid_loss, valid_score, valid_constraint = validate(args, model, val_loader, problem, mu_param)
                model.train()
                valid_losses.append(valid_loss)
                valid_scores.append(valid_score)
                valid_constraints.append(valid_constraint)
                valid_time = time.time() - valid_start_time
                
                if test_loader is not None:
                    model.eval()
                    test_loss, test_score, test_constraint = validate(args, model, test_loader, problem, mu_param)
                    model.train()
                    test_losses.append(test_loss)
                    test_scores.append(test_score)
                    test_constraints.append(test_constraint)
                else:
                    test_loss = np.inf
                    test_score = -np.inf
                    
                # print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f} valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}")
                print(f"  VALIDATION epoch={ep} steps={steps} t={valid_time:0.2f}\n\
                            valid_loss={valid_loss} valid_score={valid_score} valid_constraint={valid_constraint}\n\
                            test_loss={test_loss} test_score={test_score} test_constraint={test_constraint}")

            # occasionally save model
            # if args.save_freq != 0 and ep % args.save_freq == 0:
            #     save_model(model, f"{model_folder}/model_ep{ep}.pt")
        
        ep += 1
        

def predict(model, loader, args):
    batches = []
    # TODO decide return signature and transform.
    for batch in loader:
        assert False, "not adapted to new featurize_batch: proceed with caution!"
        x_in, edge_index, edge_weight = featurize_batch(args, batch)
        x_out = model(x_in, edge_index, edge_weight)
        batches.append((x_out, edge_index))
    return batches
