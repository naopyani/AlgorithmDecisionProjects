import cvxpy as cp
import numpy as np


def profit_max(solver_data):
    """
    线性规划算法求利润最大
    :param solver_data: 求解数据
    :return:
    """
    num_products = len(solver_data["时间成本"])
    labor_hours = np.array(solver_data["时间成本"])
    material_pounds = np.array(solver_data["时间成本"])
    product_prices = np.array(solver_data["产品售价"])
    unit_costs = np.array(solver_data["产品成本价"])
    demand = np.array(solver_data["市场需求量"])
    available_labor_hours = solver_data["可用工时"]
    available_material_pounds = solver_data["可用原材料"]
    # 定义变量
    x = cp.Variable(num_products, nonneg=True)
    # 定义目标函数
    objective = cp.Maximize(cp.sum(cp.multiply(product_prices - unit_costs, x)))
    # 定义约束条件
    constraints = [
        cp.sum(cp.multiply(labor_hours, x)) <= available_labor_hours,
        cp.sum(cp.multiply(material_pounds, x)) <= available_material_pounds,
        x <= demand
    ]
    # 定义问题并求解
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # 输出结果
    print('最优目标值:', problem.value)
    print('最优解:')
    for i in range(num_products):
        print('产品', i + 1, ':', x.value[i])


if __name__ == '__main__':
    solver_data = {
        "时间成本": [6, 5, 4, 3, 2.5, 1.5],
        "材料成本": [3.2, 2.6, 1.5, 0.8, 0.7, 0.3],
        "产品成本价": [6.5, 5.7, 3.6, 2.8, 2.2, 1.2],
        "产品售价": [12.5, 11.0, 9.0, 7.0, 6.0, 3.0],
        "市场需求量": [960, 928, 1041, 977, 1084, 1055],
        "可用工时": 4500,
        "可用原材料": 1600
    }
    profit_max(solver_data)
