import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 参数设置
np.random.seed(42)
N_samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 样本量
n_experiments = 1000      # 实验重复次数
alpha = 0.05             # 缺货概率上限 
cv_list = [0.3, 0.5]     # 变异系数
dist_types = ['normal', 'gamma', 'exponential']  # 需求分布类型

# 存储结果
results = {
    'service_level': {dist: {cv: [] for cv in cv_list} for dist in dist_types},
    'surplus': {dist: {cv: [] for cv in cv_list} for dist in dist_types}
}

def generate_demand(dist, N, a, b, x, cv):
    if dist == 'normal':
        # 线性正态需求 D = a + bx + u, u~N(0,(cv*μ)^2)
        mu_D = a + b * x[:, 1]
        sigma_D = cv * (a + b * 0.5)  # μ at mean price x=0.5
        D = mu_D + np.random.normal(0, sigma_D, N)
    elif dist == 'gamma':
        # Gamma分布：保持与正态相同的均值和CV
        mu_D = a + b * x[:, 1]
        shape = (1 / (cv ** 2)) * np.ones_like(mu_D)
        scale = (mu_D * (cv ** 2)).clip(min=1e-6)
        D = np.random.gamma(shape, scale)
    elif dist == 'exponential':
        # 非线性指数需求
        # D = (a + b*exp(x)) * ε, ε~Exp(1), 调整以实现目标CV
        mu_D = a + b * np.exp(x[:, 1])
        # 指数分布的CV=1，通过缩放调整到目标CV
        scaling_factor = cv  # 因为Exp(1)的std=1，而CV=std/mean=1/1=1
        D = mu_D * np.random.exponential(scaling_factor, N)
    return np.maximum(D, 0)  # 截断负需求

# 依次验证三种分布数据
for dist in dist_types:
    for cv in cv_list:
        for N in N_samples:
            service_levels = []
            surpluses = []
            
            for _ in range(n_experiments):
                # 生成特征数据
                x_mean, x_std = 0.5, 0.25
                x = np.random.normal(x_mean, x_std, N)
                x = np.maximum(x, 0)  # 截断负价格
                x = np.column_stack([np.ones(N), x])  # 添加截距项
                
                # 生成参数（论文参数范围）
                a = np.random.uniform(1000, 2000)
                b = np.random.uniform(-1000, -500)
                if dist == 'exponential':
                    a = np.random.uniform(3000, 4000)  # 论文特殊调整
                
                # 生成需求数据
                D = generate_demand(dist, N, a, b, x, cv)
                
                # --- SAA模型求解（论文公式7-11）---
                model = gp.Model("SAA_Newsvendor")
                r = model.addVars(2, lb=-GRB.INFINITY, name="r")
                y = model.addVars(N, lb=0, name="y")
                gamma = model.addVars(N, vtype=GRB.BINARY, name="gamma")
                
                model.setObjective(gp.quicksum(y[i] for i in range(N)), GRB.MINIMIZE)
                
                M = 1e6
                for i in range(N):
                    model.addConstr(y[i] >= r[0] + r[1]*x[i,1] - D[i])
                    model.addConstr(r[0] + r[1]*x[i,1] + M*gamma[i] >= D[i])
                model.addConstr(gp.quicksum(gamma[i] for i in range(N)) <= alpha*N)
                
                model.Params.OutputFlag = 0
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    r_opt = [r[0].X, r[1].X]
                    # 生成测试数据（10^6样本）
                    x_test = np.random.normal(x_mean, x_std, 10**6)
                    x_test = np.maximum(x_test, 0)
                    x_test = np.column_stack([np.ones(10**6), x_test])
                    D_test = generate_demand(dist, 10**6, a, b, x_test, cv)
                    
                    # 计算指标
                    I_test = r_opt[0] + r_opt[1] * x_test[:,1]
                    service_level = np.mean(I_test >= D_test)
                    surplus = np.mean(np.where(I_test > D_test, I_test - D_test, 0))
                    
                    service_levels.append(service_level)
                    surpluses.append(surplus)
            
            # 保存结果
            avg_service = np.mean(service_levels)
            avg_surplus = np.mean(surpluses)
            results['service_level'][dist][cv].append(avg_service)
            results['surplus'][dist][cv].append(avg_surplus)
            print(f"[{dist.upper()}][CV={cv}] N={N}: Service Level={avg_service:.3f}, Surplus={avg_surplus:.1f}")

# 绘制图表展示结果
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for i, dist in enumerate(dist_types):
    for j, cv in enumerate(cv_list):
        ax = axes[i,j]
        ax.plot(N_samples, results['service_level'][dist][cv], 'o-', color='C0')
        ax.axhline(1-alpha, color='r', linestyle='--')
        ax.set_title(f'{dist.capitalize()} Demand, CV={cv}')
        ax.set_ylabel('Service Level')
        if i == 2: ax.set_xlabel('Sample Size (N)')
        
        ax2 = ax.twinx()
        ax2.plot(N_samples, results['surplus'][dist][cv], 's--', color='C1')
        ax2.set_ylabel('Avg. Surplus Inventory')

plt.tight_layout()
plt.savefig('saa_results_final_corrected.png', dpi=300)
plt.show()