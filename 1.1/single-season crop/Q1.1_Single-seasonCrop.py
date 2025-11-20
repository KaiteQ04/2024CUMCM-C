import pandas as pd
from ortools.linear_solver import pywraplp

# 文件路径
crop_types_path = r'D:\数模\C题\数据预处理\第一题第一问\作物种类.xlsx'
sales_results_path = r'D:\数模\C题\数据预处理\第一题第一问\连接结果.xlsx'
analysis_path = r'D:\数模\C题\数据预处理\第一题第一问\连接结果.xlsx'
suitable_plots_path = r'D:\数模\C题\数据预处理\第一题第一问\每种作物适合种植的地块类型与季别_合并.xlsx'
output_path = r'D:\数模\C题\数据预处理\第一题第一问\第一季单季作物.xlsx'

# 读取Excel文件
crop_types_df = pd.read_excel(crop_types_path, sheet_name='第一季（单）')
sales_results_df = pd.read_excel(sales_results_path, sheet_name='2023销售结果')
analysis_df = pd.read_excel(analysis_path, sheet_name='销售结果分析')
suitable_plots_df = pd.read_excel(suitable_plots_path)

# 获取地块信息
plots = {
    '平旱地': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
    '梯田': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14'],
    '山坡地': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
}
areas = {
    'A1': 80, 'A2': 55, 'A3': 35, 'A4': 72, 'A5': 68, 'A6': 55,
    'B1': 60, 'B2': 46, 'B3': 40, 'B4': 28, 'B5': 25,
    'B6': 86, 'B7': 55, 'B8': 44, 'B9': 50, 'B10': 25,
    'B11': 60, 'B12': 45, 'B13': 35, 'B14': 20,
    'C1': 15, 'C2': 13, 'C3': 15, 'C4': 18, 'C5': 27, 'C6': 20
}

# 提取作物相关数据
crops = list(set(crop_types_df['作物名称']))
crop_info = {}
for index, row in analysis_df.iterrows():
    crop_info[row['作物名称']] = {
        '产量': row['作物产量（销量）'],
        '价格': row['实际销售价格'],
        '成本': row['种植成本/(元/亩)'],
        '地块': row['地块名称'],
        '地块类型': row['地块类型']
    }
    # 标记豆类作物
    if '豆类' in row['作物类型']:
        crop_info[row['作物名称']]['is_bean'] = True
    else:
        crop_info[row['作物名称']]['is_bean'] = False
# 创建线性规划模型
solver = pywraplp.Solver.CreateSolver('SCIP')

# 定义变量
variables = {}
for year in range(2024, 2031):
    for plot_type, plots_list in plots.items():
        for plot in plots_list:
            for crop in crops:
                # 使用布尔索引
                if len(suitable_plots_df.loc[suitable_plots_df['作物名称'] == crop, plot_type]) > 0 and \
                   suitable_plots_df.loc[suitable_plots_df['作物名称'] == crop, plot_type].iloc[0] == 1:
                    variables[(year, plot, crop)] = solver.NumVar(0, areas[plot], f'x_{year}_{plot}_{crop}')
            
binary_vars = {}

for year in range(2024, 2031):
    for plot_type, plots_list in plots.items():
        for plot in plots_list:
            for crop in crops:
                # 使用布尔索引
                if len(suitable_plots_df.loc[suitable_plots_df['作物名称'] == crop, plot_type]) > 0 and \
                   suitable_plots_df.loc[suitable_plots_df['作物名称'] == crop, plot_type].iloc[0] == 1:
                    variables[(year, plot, crop)] = solver.NumVar(0, areas[plot], f'x_{year}_{plot}_{crop}')
                    binary_vars[(year, plot, crop)] = solver.BoolVar(f'y_{year}_{plot}_{crop}')


# 添加约束条件
# 每个地块总种植面积不得超过该地块面积
for year in range(2024, 2031):
    for plot in areas.keys():
        solver.Add(sum(variables.get((year, plot, crop), 0) for crop in crops) <= areas[plot])

# 每种作物在同一地块不能连续重茬种植
for year in range(2024, 2030):  # 不考虑最后一年的重茬
    for plot in areas.keys():
        for crop in crops:
            solver.Add(variables.get((year, plot, crop), 0) + variables.get((year+1, plot, crop), 0) <= areas[plot])

# 每个地块的所有土地三年内至少种植一次豆类作物
for plot in areas.keys():
    solver.Add(sum(variables.get((year, plot, crop), 0) * crop_info[crop]['is_bean'] for year in range(2024, 2027) for crop in crops if 'is_bean' in crop_info[crop]) >= 1)

# 每种作物每季的种植地不能太分散
# 假设作物连续地块为 A1 到 A6
#for year in range(2024, 2031):
#   for crop in crops:
#        for plot_type, plots_list in plots.items():
#            for i in range(len(plots_list) - 1):
#                solver.Add(sum(variables.get((year, plots_list[i], crop), 0)) + sum(variables.get((year, plots_list[i+1], crop), 0)) <= areas[plots_list[i]])
# 每种作物每季的种植地必须连续
for year in range(2024, 2031):
    for crop in crops:
        for plot_type, plots_list in plots.items():
            for i in range(len(plots_list) - 1):
                # 如果作物在当前地块和下一个地块上都有种植，则它们的总面积不得超过当前地块的面积
                solver.Add(
                    variables.get((year, plots_list[i], crop), 0) + 
                    variables.get((year, plots_list[i+1], crop), 0) <= 
                    areas[plots_list[i]]
                )

# 设置目标函数
objective = solver.Objective()
for year in range(2024, 2031):
    for plot in areas.keys():
        for crop in crops:
            if crop in crop_info:
                profit_per_acre = (crop_info[crop]['价格'] * crop_info[crop]['产量']) - crop_info[crop]['成本']
                var = variables.get((year, plot, crop))
                if var is not None:
                    objective.SetCoefficient(var, profit_per_acre)

objective.SetMaximization()

# 求解模型
status = solver.Solve()

results = []
if status == pywraplp.Solver.OPTIMAL:
    print('Optimal solution found.')
    # 输出最优解
    for key, var in variables.items():
        if var.solution_value() > 0:
            results.append({
                '年份': key[0],
                '地块': key[1],
                '作物': key[2],
                '种植面积': var.solution_value()
            })
else:
    print('No optimal solution found.')

# 将结果保存到Excel文件
results_df = pd.DataFrame(results)
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='第一季单季作物', index=False)

print(f'Results saved to {output_path}')