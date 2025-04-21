import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 数据生成
N = 100  # Sample size
d = 2    # Dimension of variable r
M = 1e5  # Big M constant
alpha = 0.1  # Parameter
theta = (1/N)**(1/d)  # Wasserstein distance parameter
cv = 0.3  # Coefficient of variation (0.3 or 0.5)

# 参数生成
a = np.random.uniform(1000, 2000)  # 均匀分布
b = np.random.uniform(-1000, -500) # 均匀分布

# 生成 x_i ~ N(0.5, 0.25^2) （训练集价格数值）
x_opt = np.random.normal(loc=0.5, scale=0.25, size=(N, d))
x_bar_opt = np.mean(x_opt)

# 计算 D_i （训练集需求数值）
D_means_opt = a + b * np.sum(x_opt, axis=1)  #训练集需求 D_i均值
D_std_opt = cv * (a + b * x_bar_opt)        # 计算训练集需求标准差
D_opt = np.random.normal(loc=D_means_opt, scale=D_std_opt, size=N)

# 模型命名
model = gp.Model("Wasserstein_DRO")

# 定义变量
# 连续性变量定义
r = model.addVars(d, lb=-GRB.INFINITY, name="r")  # Unbounded
y = model.addVars(N, lb=0.0, name="y")           # Non-negative
s = model.addVars(N, lb=0.0, name="s")           # Non-negative
t = model.addVar(lb=-GRB.INFINITY, name="t")      # Unbounded
z = model.addVar(lb=1.0, name="z")                # z ≥ 1

# 0-1变量命名
q = model.addVars(N, vtype=GRB.BINARY, name="q")  # q_i ∈ {0,1}

# 目标函数
model.setObjective(gp.quicksum(y[i] for i in range(N)), GRB.MINIMIZE)

# 用训练集数据构造约束条件
# Constraint 1: (1/N)*sum(s_j) + theta*z <= alpha*t
model.addConstr(
    (1/N) * gp.quicksum(s[j] for j in range(N)) + theta * z <= alpha * t,
    name="wasserstein_constraint"
)

# Constraint 2: |r_j| <= z → -z <= r_j <= z
for j in range(d):
    model.addConstr(r[j] <= z, name=f"r_pos_{j}")
    model.addConstr(r[j] >= -z, name=f"r_neg_{j}")

# Constraint 3: r^T x_i - D_i + M q_i >= t - s_i
for i in range(N):
    model.addConstr(
        gp.quicksum(r[j] * x_opt[i,j] for j in range(d)) - D_opt[i] + M * q[i] >= t - s[i],
        name=f"bigM_1_{i}"
    )

# Constraint 4: M*(1 - q_i) >= t - s_i
for i in range(N):
    model.addConstr(
        M * (1 - q[i]) >= t - s[i],
        name=f"bigM_2_{i}"
    )

# Constraint 5: y_i >= r^T x_i - D_i
for i in range(N):
    model.addConstr(
        y[i] >= gp.quicksum(r[j] * x_opt[i,j] for j in range(d)) - D_opt[i],
        name=f"y_lower_{i}"
    )

#求解测试集最优解及其r向量
try:
    # Parameter settings
    model.setParam('OutputFlag', 1)      # Show solving process
    model.setParam('MIPGap', 1e-4)       # MIP tolerance gap
    model.setParam('FeasibilityTol', 1e-6) # Feasibility tolerance
    model.setParam('IntFeasTol', 1e-5)   # Integer feasibility tolerance
    model.setParam('NumericFocus', 1)    # Improve numerical stability

    # 求解
    model.optimize()

    # 输出模型求解状态
    if model.status == GRB.OPTIMAL:
        print("\nOptimization successful!")
        print(f"Optimal objective value: {model.objVal:.4f}")
        
        # Get variable values
        r_val = np.array([r[j].X for j in range(d)])
        t_val = t.X
        z_val = z.X
        
        print(f"Optimal r = {np.round(r_val, 4)}")
        print(f"Optimal t = {t_val:.4f}")
        print(f"Optimal z = {z_val:.4f}")
        
        # 生成测试集验证模型
        print("\nStarting evaluation on 1000 independent test sets...")
        
        # 初始化目标指标集合
        service_levels = []  # Proportion where r'x_j - D_j <= t
        non_negative_ratios = []  # Proportion where r'x_j - D_j >= 0
        avg_inventories = []  # Average of max(r'x_j - D_j, 0)
        sum_residuals = []  # Sum of max(r'x_j - D_j, 0)
        
        for _ in range(1000):
            # 按照相同的参数再生成1000组数据构造测试集
            x_test = np.random.normal(loc=0.5, scale=0.25, size=(N, d))
            x_bar_test = np.mean(x_test)
            
            # 保证参数相同
            D_means_test = a + b * np.sum(x_test, axis=1)
            D_std_test = cv * (a + b * x_bar_test)
            D_test = np.random.normal(loc=D_means_test, scale=D_std_test, size=N)
            
            # 计算 r^T x_i - D_i 
            residuals = np.dot(x_test, r_val) - D_test
            
            # 计算服务水平 (满足 r'x_j - D_j <= t的比例)
            service_level = np.mean(residuals <= t_val)
            service_levels.append(service_level)
            
            # 计算剩余库存非负的比例 (满足 r'x_j - D_j >= 0的比例)
            non_negative_ratio = np.mean(residuals >= 0)
            non_negative_ratios.append(non_negative_ratio)
            
            # 截断非负值 (max(r'x_j - D_j, 0))
            y_test = np.maximum(residuals, 0)
            
            # 计算剩余库存均值
            avg_inventory = np.mean(y_test)
            avg_inventories.append(avg_inventory)
            
            # 计算各批次剩余库存之和(sum of max(r'x_j - D_j, 0))
            sum_residual = np.sum(y_test)
            sum_residuals.append(sum_residual)
        
        # 计算并输出结果
        print("\nEvaluation results:")
        
        # 服务水平数据
        print("\nService Level (proportion where r'x_j - D_j <= t):")
        print(f"服务水平均值: {np.mean(service_levels):.4f} ± {np.std(service_levels):.4f}")
        print(f"服务水平最小值: {np.min(service_levels):.4f}")
        print(f"服务水平最大值: {np.max(service_levels):.4f}")
        
        
        # 剩余库存数据
        print("\nRemaining Inventory Statistics:")
        print(f"样本点剩余库存均值: {np.mean(avg_inventories):.4f} ± {np.std(avg_inventories):.4f}")
        print(f"各批次剩余库存均值(本文关注指标): {np.mean(sum_residuals):.4f} ± {np.std(sum_residuals):.4f}")
        
        # 输出分位数便于突出异常值影响
        print("\nService Level Percentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"{p}th percentile: {np.percentile(service_levels, p):.4f}")
            
        print("\nAverage Inventory Percentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"{p}th percentile: {np.percentile(avg_inventories, p):.4f}")
        
    else:
        print(f"\nOptimization terminated with status: {model.status}")
        if model.status == GRB.INFEASIBLE:
            print("Suggested checks:")
            print("1. Data ranges (especially D_i values)")
            print("2. Try adjusting alpha parameter")
            print("3. Check Big M value")
        elif model.status == GRB.UNBOUNDED:
            print("Problem is unbounded - check objective and constraints")
        elif model.status == GRB.TIME_LIMIT:
            print("Time limit reached - try increasing time limit")

except gp.GurobiError as e:
    print(f"Gurobi error: {str(e)}")
except Exception as e:
    print(f"Runtime error: {str(e)}")
finally:
    # Release resources
    model.dispose()