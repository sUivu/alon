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
    # 记录开始时间（使用高精度计时器）
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

    # 计算总耗时（秒）
    elapsed_seconds = end_time - start_time

    # 转换为时、分、秒
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60

    # 输出结果（保留两位小数，精确到百分之一秒）
    print(f"程序运行总时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    # 3. 获取问题类型 (如 VertexCoverProblem)
    
    
    # problem_cls = get_problem(args)
    
    # print(f"开始使用 Gurobi 求解数据集: {args.dataset}, 问题: {args.problem_type}")
    # print(f"测试集大小: {len(test_loader.dataset)}")

    # results = []
    # runtimes = []
    
    # # 4. 逐个图进行求解
    # # Gurobi 只能逐个处理图，所以 batch_size 建议设为 1 或者手动拆解 batch
    # for i, batch in enumerate(test_loader):
    #     # 将 Batch 拆解为单个图对象
    #     examples = batch.to_data_list()
    #     for j, example in enumerate(examples):
    #         start_t = time.time()
            
    #         # 调用你 problem 类中的 gurobi 接口
    #         # 返回: x_vals (解向量), status (求解状态), runtime (求解耗时)
    #         x_vals, status, runtime = problem_cls.gurobi(args, example)
            
    #         # 计算该解对应的 Score (对于 MVC 来说是 -选中顶点数)
    #         # 注意：Gurobi 返回的是 np.array，需要转成 torch 传给 score 函数
    #         score = problem_cls.score(args, torch.from_numpy(x_vals).float(), example)
            
    #         results.append(score)
    #         runtimes.append(runtime)
            
    #         if (len(results)) % 10 == 0:
    #             print(f"已完成: {len(results)} 张图, 当前平均 Score: {np.mean(results):.4f}")
    # # 5. 输出最终统计结果
    # print("\n" + "="*30)
    # print(f"Gurobi 最终统计结果 ({args.problem_type}):")
    # print(f"平均最优值 (Score): {np.mean(results):.4f}")
    # print(f"标准差: {np.std(results):.4f}")
    # print(f"平均求解耗时: {np.mean(runtimes):.4f}s")
    # print("="*30)