from scipy.optimize import minimize
import numpy as np
from sympy import sympify, Pow, symbols
import re


def goal_seek(formula_str: str, target: float):
    """
    单变量求解
    :param formula_str:原函数算法公式，需要包含自变量x，如 "(58+70+x)/3"
    :param target: 需要优化的目标值
    :return: 返回自变量最优解
    """
    letters = re.findall(r'[a-zA-Z]', formula_str)
    if len(letters) == 0:
        raise RuntimeError("未找到自变量")
    if len(letters) != 1:
        raise RuntimeError("单变量求解，找到多个自变量：" + str(letters))
    x = symbols(letters[0])
    expr = sympify(formula_str)
    squared_diff = Pow(expr - target, 2)
    solve_result = minimize(lambda y: squared_diff.subs(x, y[0]), np.array(0), method='BFGS')
    return round(float(solve_result.x), 2)


if __name__ == '__main__':
    input_str = "(58+70+72+60+x)/5"
    solve_x = goal_seek(input_str, 72)
    print(solve_x)
