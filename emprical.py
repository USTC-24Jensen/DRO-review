import numpy as np
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
import warnings

class DataGenerator:
    @staticmethod
    def generate_batch(sample_size, cv=0.5, batch_size=100):
        """生成具有相同分布参数的批量合成数据"""
        num_batches = (sample_size + batch_size - 1) // batch_size
        a_values = np.random.uniform(1000, 2000, size=num_batches)
        b_values = np.random.uniform(-1000, -500, size=num_batches)
        
        xi_all = []
        Di_all = []
        a_all = []
        b_all = []
        x_means = []
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, sample_size - i * batch_size)
            xi = np.random.normal(loc=0.5, scale=0.25, size=current_batch_size)
            xi = np.where(xi < 0, 0, xi)
            x_mean = np.mean(xi)
            
            a = a_values[i]
            b = b_values[i]
            mean_Di = a + b * xi
            std_Di = cv * (a + b * x_mean)
            std_Di = np.abs(std_Di)
            Di = np.random.normal(loc=mean_Di, scale=std_Di)
            Di = np.where(Di < 0, 0, Di)
            
            xi_all.extend(xi)
            Di_all.extend(Di)
            a_all.extend([a] * current_batch_size)
            b_all.extend([b] * current_batch_size)
            x_means.append(x_mean)
        
        return np.array(xi_all), np.array(Di_all), a_all, b_all, np.mean(x_means)

class ModelTrainer:
    def __init__(self, alpha, big_M):
        self.alpha = alpha
        self.big_M = big_M
        
    def train(self, xi_train, Di_train):
        """训练优化模型"""
        try:
            model = gp.Model("Training_Phase")
            N_train = len(xi_train)
            
            r = model.addVars(N_train, lb=-GRB.INFINITY, name="r")
            y = model.addVars(N_train, lb=0, name="y")
            gamma = model.addVars(N_train, vtype=GRB.BINARY, name="gamma")
            
            model.setObjective(gp.quicksum(y[i] for i in range(N_train)), GRB.MINIMIZE)
            
            for i in range(N_train):
                model.addConstr(y[i] >= r[i] * xi_train[i] - Di_train[i], f"constr1_{i}")
            
            model.addConstr(
                gp.quicksum(gamma[i] for i in range(N_train)) >= (1 - self.alpha) * N_train, 
                "gamma_sum"
            )
            
            for i in range(N_train):
                model.addGenConstrIndicator(
                    gamma[i], True, 
                    Di_train[i] - r[i] * xi_train[i] <= 0
                )
                model.addGenConstrIndicator(
                    gamma[i], False, 
                    Di_train[i] - r[i] * xi_train[i] >= 0
                )
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return self._collect_results(model, r, xi_train, Di_train)
            return {'success': False}
            
        except Exception as e:
            print(f"训练失败: {str(e)}")
            return {'success': False}
    
    def _collect_results(self, model, r, xi_train, Di_train):
        """收集训练模型结果"""
        r_opt = np.array([r[i].X for i in range(len(xi_train))])
        rx_mean = np.mean(r_opt * xi_train)
        Di_mean = np.mean(Di_train)
        satisfy_ratio = np.mean(Di_train <= r_opt * xi_train)
        
        return {
            'rx_mean': rx_mean,
            'Di_mean': Di_mean,
            'satisfy_ratio': satisfy_ratio,
            'r_opt': r_opt,
            'xi_train': xi_train,
            'Di_train': Di_train,
            'success': True
        }

class ModelEvaluator:
    @staticmethod
    def evaluate(train_result, N_test, cv_train, batch_size=100):
        """在测试数据上评估训练模型"""
        r_opt = train_result['r_opt']
        N_train = len(r_opt)
        
        xj_test, Dj_test, _, _, _ = DataGenerator.generate_batch(N_test, cv_train, batch_size)
        r_test = np.tile(r_opt, (N_test // N_train + 1))[:N_test] if N_test > N_train else r_opt[:N_test]
        
        # 计算原始差值（截断前）
        raw_diff = r_test * xj_test - Dj_test
        
        # 异常值处理（基于IQR方法）
        q1, q3 = np.percentile(raw_diff, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 标记异常值（但不删除，后续会截断为0）
        is_outlier = (raw_diff < lower_bound) | (raw_diff > upper_bound)
        
        # 截断处理（负值和异常值都设为0）
        y_j = np.where((raw_diff < 0) | is_outlier, 0, raw_diff)
        
        return {
            'y_values': y_j,
            'raw_diff': raw_diff,  # 保留原始差值用于分析
            'outlier_count': np.sum(is_outlier),
            'non_neg_count': np.sum(y_j > 0),
            'total_sum': np.sum(y_j),
            'total_count': len(y_j),
            'success': True
        }

class ExperimentRunner:
    def __init__(self, params):
        self.params = params
        self._setup()
        
    def _setup(self):
        """初始化参数并抑制警告"""
        gp.setParam('OutputFlag', 0)
        warnings.filterwarnings('ignore')
        
    def run(self):
        """运行完整实验"""
        results = {
            'rx_means': [],
            'Di_means': [],
            'satisfy_ratios': [],
            'all_y_values': [],
            'all_raw_diff': [],  # 存储原始差值
            'total_outliers': 0,
            'total_non_neg': 0,
            'total_sum': 0,
            'total_points': 0,
            'successful_experiments': 0
        }
        
        print(f"=== 开始 {self.params['num_experiments']} 次实验 ===")
        pbar = tqdm(total=self.params['num_experiments'], desc="实验进度")
        
        trainer = ModelTrainer(self.params['alpha'], self.params['M'])
        
        for _ in range(self.params['num_experiments']):
            xi_train, Di_train, _, _, _ = DataGenerator.generate_batch(
                self.params['N_train'], 
                self.params['cv_train'],
                self.params.get('batch_size', 100)
            )
            train_result = trainer.train(xi_train, Di_train)
            
            if train_result['success']:
                results['successful_experiments'] += 1
                results['rx_means'].append(train_result['rx_mean'])
                results['Di_means'].append(train_result['Di_mean'])
                results['satisfy_ratios'].append(train_result['satisfy_ratio'])
                
                eval_result = ModelEvaluator.evaluate(
                    train_result,
                    self.params['N_test'],
                    self.params['cv_train'],
                    self.params.get('batch_size', 100)
                )
                
                # 累积统计量
                results['all_y_values'].extend(eval_result['y_values'])
                results['all_raw_diff'].extend(eval_result['raw_diff'])
                results['total_outliers'] += eval_result['outlier_count']
                results['total_non_neg'] += eval_result['non_neg_count']
                results['total_sum'] += eval_result['total_sum']
                results['total_points'] += eval_result['total_count']
            
            pbar.update(1)
            current_mean = results['total_sum'] / results['total_points'] if results['total_points'] > 0 else 0
            pbar.set_postfix({
                '成功实验': results['successful_experiments'],
                '实时均值': f"{current_mean:.2f}",
                '异常值%': f"{results['total_outliers']/results['total_points']*100:.1f}%" if results['total_points'] > 0 else "0%"
            })
        
        pbar.close()
        self._print_summary(results)
    
    def _print_summary(self, results):
        """打印实验摘要"""
        print("\n=== 实验完成 ===")
        print(f"成功实验次数: {results['successful_experiments']}/{self.params['num_experiments']}")
        
        if results['successful_experiments'] > 0:
            self._print_training_results(results)
            self._print_evaluation_results(results)
            self._print_outlier_analysis(results)  # 新增异常值分析
        else:
            print("\n警告: 所有实验均失败，无有效结果")
    
    def _print_training_results(self, results):
        """打印训练阶段结果"""
        print("\n=== 训练阶段统计 ===")
        print(f"r*x_i 均值 ({results['successful_experiments']} 次实验): "
              f"{np.mean(results['rx_means']):.2f} ± {np.std(results['rx_means']):.2f}")
        print(f"D_i 均值 ({results['successful_experiments']} 次实验): "
              f"{np.mean(results['Di_means']):.2f} ± {np.std(results['Di_means']):.2f}")
        print(f"满足比例 ({results['successful_experiments']} 次实验): "
              f"{np.mean(results['satisfy_ratios']):.4f} ± {np.std(results['satisfy_ratios']):.4f}")
    
    def _print_evaluation_results(self, results):
        """打印评估阶段结果"""
        print("\n=== 测试阶段统计 ===")
        print(f"总测试样本点数: {results['total_points']}")
        
        trunc_mean = results['total_sum'] / results['total_points'] if results['total_points'] > 0 else 0
        non_neg_proportion = results['total_non_neg'] / results['total_points'] if results['total_points'] > 0 else 0
        
        print(f"截断后均值（含0值）: {trunc_mean:.2f}")
        print(f"严格正值比例: {non_neg_proportion:.4f}")
        print(f"全局 yj 统计 - 最小值: {np.min(results['all_y_values']):.2f}, 最大值: {np.max(results['all_y_values']):.2f}")
    
    def _print_outlier_analysis(self, results):
        """新增：异常值分析"""
        print("\n=== 异常值分析 ===")
        print(f"总异常值数量: {results['total_outliers']} ({results['total_outliers']/results['total_points']*100:.2f}%)")
        
        # 分析异常值分布
        raw_diff = np.array(results['all_raw_diff'])
        outliers = raw_diff[(raw_diff < np.percentile(raw_diff, 25) - 1.5*(np.percentile(raw_diff, 75)-np.percentile(raw_diff, 25))) | 
                          (raw_diff > np.percentile(raw_diff, 75) + 1.5*(np.percentile(raw_diff, 75)-np.percentile(raw_diff, 25)))]
        
        if len(outliers) > 0:
            print(f"异常值最小值: {np.min(outliers):.2f}, 最大值: {np.max(outliers):.2f}")
            print(f"异常值中位数: {np.median(outliers):.2f}")
        else:
            print("未检测到异常值")

if __name__ == "__main__":
    # 实验参数
    params = {
        'N_train': 100,
        'N_test': 100,
        'cv_train': 0.5,
        'alpha': 0.05,
        'M': 1e7,
        'num_experiments': 1000,
        'batch_size': 100
    }
    
    # 运行实验
    runner = ExperimentRunner(params)
    runner.run()