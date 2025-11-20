import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms

# 读取Excel文件
sales_results_path = r'D:\数模\C题\第二问草稿\单季\使用数据\连接结果.xlsx'
crop_types_path = r'D:\数模\C题\第二问草稿\单季\使用数据\每种作物适合种植的地块类型与季别_合并.xlsx'
crop_varieties_path = r'D:\数模\C题\第二问草稿\单季\使用数据\作物种类.xlsx'

sales_results_df = pd.read_excel(sales_results_path, sheet_name='2023销售结果')
analysis_df = pd.read_excel(sales_results_path, sheet_name='销售结果分析')
crop_types_df = pd.read_excel(crop_types_path)
crop_varieties_df = pd.read_excel(crop_varieties_path, sheet_name='第一季（单）')

# 读取 Excel 文件中的数据
file_path = r"D:\数模\C题\第二问草稿\单季\使用数据\作物种类.xlsx"
sheet_name = "第一季（单）"

crop_varieties_df = pd.read_excel(file_path, sheet_name=sheet_name)

# 生成 valid_crops 字典，存储有效的作物及其是否为豆类
valid_crops = {}  # 存储有效的作物及其是否为豆类
for index, row in crop_varieties_df.iterrows():
    crop_name = row['作物名称']
    crop_type = row['作物类型']
    is_legume = '豆类' in crop_type
    valid_crops[crop_name] = is_legume

# 2023年的实际种植数据
actual_planting_2023 = {}

# 定义需要保留的字段列表
fields_to_keep = ['作物名称', '作物类型', '地块名称', '亩产量/斤', '地块类型', '种植成本/(元/斤)', 
                 '种植面积/亩', '作物产量（销量）', '实际销售价格']

# 遍历2023年的销售结果数据
for index, row in sales_results_df.iterrows():
    plot_name = row['地块名称']
    crop_name = row['作物名称']
    
    # 查找作物类型和其他相关信息
    crop_info = sales_results_df[sales_results_df['作物名称'] == crop_name]
    
    if not crop_info.empty:
        # 获取作物类型
        crop_type = crop_info.iloc[0]['作物类型']
        
        # 检查是否为豆类
        is_legume = '豆类' in crop_type
        
        # 获取其他相关信息
        for field in fields_to_keep:
            value = row[field] if field != '作物类型' else crop_type
            setattr(row, f'field_{field}', value)  # 使用setattr动态设置属性以便后续访问
        
        # 存储到字典中
        actual_planting_2023[(plot_name, crop_name)] = {f'field_{field}': getattr(row, f'field_{field}') for field in fields_to_keep}


# 提取地块面积
land_areas = {
    '平旱地': {'A1': 80, 'A2': 55, 'A3': 35, 'A4': 72, 'A5': 68, 'A6': 55},
    '梯田': {f'B{i+1}': 60 if i == 0 else 46 if i == 1 else 40 if i == 2 else 28 if i == 3 else 25 if i == 4 else 86 if i == 5 else 55 if i == 6 else 44 if i == 7 else 50 if i == 8 else 25 if i == 9 else 60 if i == 10 else 45 if i == 11 else 35 if i == 12 else 20 for i in range(14)},
    '山坡地': {'C1': 15, 'C2': 13, 'C3': 15, 'C4': 18, 'C5': 27, 'C6': 20}
}

# 不确定性信息
sales_growth_rate = {
    '小麦': (0.05, 0.10),
    '玉米': (0.05, 0.10),
    '其他': 0.05,
}
yield_variation = 0.10
cost_growth_rate = 0.05
price_growth_rate = {
    '粮食类': 0.00,
    '蔬菜类': 0.05,
    '食用菌': (-0.05, -0.01),
    '羊肚菌': -0.05,
}

# 蒙特卡洛模拟
def monte_carlo_simulation(actual_data, years):
    simulated_data = []
    for year in range(years):
        temp_data = {}
        for (plot, crop), data in actual_data.items():
            # 销售量
            if crop in ['小麦', '玉米']:
                sale = data['field_作物产量（销量）'] * (1 + np.random.uniform(sales_growth_rate[crop][0], sales_growth_rate[crop][1]) * year)
            else:
                sale = data['field_作物产量（销量）'] * (1 + np.random.uniform(-sales_growth_rate['其他'], sales_growth_rate['其他']))
            
            # 亩产量
            yield_per_acre = data['field_亩产量/斤'] * (1 + np.random.uniform(-yield_variation, yield_variation))
            
            # 种植成本
            cost_per_acre = data['field_种植成本/(元/斤)'] * (1 + cost_growth_rate * year)
            
            # 销售价格
            if data['field_作物类型'] == '粮食类':
                selling_price = data['field_实际销售价格'] * (1 + price_growth_rate['粮食类'] * year)
            elif data['field_作物类型'] == '蔬菜类':
                selling_price = data['field_实际销售价格'] * (1 + price_growth_rate['蔬菜类'] * year)
            elif data['field_作物类型'] == '食用菌':
                if crop == '羊肚菌':
                    selling_price = data['field_实际销售价格'] * (1 + price_growth_rate['羊肚菌'] * year)
                else:
                    selling_price = data['field_实际销售价格'] * (1 + np.random.uniform(price_growth_rate['食用菌'][0], price_growth_rate['食用菌'][1]))
            else:
                # 处理未分类的作物类型
                selling_price = data['field_实际销售价格'] * (1 + np.random.uniform(-0.05, 0.05) * year)
            
            temp_data[crop] = {
                'yield_per_acre': yield_per_acre,
                'selling_price': selling_price,
                'cost_per_acre': cost_per_acre,
            }
        simulated_data.append(temp_data)
    return simulated_data

# 提取作物相关数据
simulated_data = monte_carlo_simulation(actual_planting_2023, 7)

# 生成多种情景
def generate_scenarios(simulated_data, years):
    scenarios = []
    for year in range(len(years)):
        temp_data = {}
        for crop, values in simulated_data[year].items():
            # 预期销售量
            if crop in ['小麦', '玉米']:
                temp_data[crop] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(sales_growth_rate[crop][0], sales_growth_rate[crop][1])),
                    'selling_price': values['selling_price'] * (1 + np.random.uniform(price_growth_rate['粮食类'], price_growth_rate['粮食类'])),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(cost_growth_rate, cost_growth_rate)),
                }
            elif crop == '羊肚菌':
                temp_data[crop] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(-yield_variation, yield_variation)),
                    'selling_price': values['selling_price'] * (1 - np.random.uniform(price_growth_rate['羊肚菌'], price_growth_rate['羊肚菌'])),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(cost_growth_rate, cost_growth_rate)),
                }
            else:
                temp_data[crop] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(-yield_variation, yield_variation)),
                    'selling_price': values['selling_price'] * (1 + np.random.uniform(price_growth_rate['食用菌'][0], price_growth_rate['食用菌'][1])),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(cost_growth_rate, cost_growth_rate)),
                }
        scenarios.append(temp_data)
    return scenarios

# 生成情景
scenarios = generate_scenarios(simulated_data, range(2024, 2031))

# 初始化DEAP环境
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 计算个体长度
num_plots = sum([len(plots) for plots in land_areas.values()])
num_years = len(scenarios)
num_crops = len(valid_crops)
individual_length = num_plots * num_years * num_crops

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=individual_length)

# 检查约束条件
def check_constraints(decoded_individual, land_areas, valid_crops, actual_planting_2023):
    # 耕地面积限制
    for year in range(len(scenarios)):
        # 创建一个字典来存储每个地块的总种植面积
        plot_total_areas = {}
        
        for land_type, plots in land_areas.items():
            for plot, area in plots.items():
                # 计算每个地块的总种植面积
                if plot not in plot_total_areas:
                    plot_total_areas[plot] = 0
                
                for crop in valid_crops:
                    plot_total_areas[plot] += decoded_individual.get((land_type, plot, crop, year), 0)
                
                # 检查是否超出面积限制
                if plot_total_areas[plot] > area:
                    return False
                
    # 不重茬约束
    crops = [None] * len(scenarios)
    for year in range(len(scenarios)):
        for crop in valid_crops:
          if decoded_individual[(land_type, plot, crop, year)]:
              crops[year] = crop
              break
        for year in range(len(scenarios) - 1):
                if crops[year] == crops[year + 1]:
                    return False

            # 豆类作物三年内种植一次的约束
    legume_years = [year for year in range(len(scenarios)) if any(valid_crops[crop] for crop in valid_crops if decoded_individual[(land_type, plot, crop, year)])]
    if len(legume_years) > 0 and max(legume_years) - min(legume_years) > 2:
           return False

    # 每种作物每季的种植地不能太分散
    for crop in valid_crops:
        for year in range(len(scenarios)):
            plots_per_type = {}
            for land_type, plots in land_areas.items():
                planted_plots = [plot for plot in plots if decoded_individual[(land_type, plot, crop, year)]]
                if planted_plots:
                    plots_per_type[land_type] = planted_plots
            
            for land_type, planted_plots in plots_per_type.items():
                if len(planted_plots) > 1:
                    if not is_consecutive(planted_plots):
                        return False
    return True


# 适应度函数
def evaluate(individual, scenarios, land_areas, valid_crops, actual_planting_2023):
    total_profit = 0
    decoded_individual = decode_individual(individual, num_years, land_areas, valid_crops)
    
    # 检查约束条件
    if not check_constraints(decoded_individual, land_areas, valid_crops, actual_planting_2023):
        return -1,
    
    # 计算总利润
    for year in range(len(scenarios)):
        for land_type, plots in land_areas.items():
            for plot, area in plots.items():
                for crop, data in scenarios[year].items():
                    if decoded_individual[(land_type, plot, crop, year)] > 0:
                        profit = (data['yield_per_acre'] * data['selling_price'] - data['cost_per_acre']*data['yield_per_acre']) * decoded_individual[(land_type, plot, crop, year)]
                        total_profit += profit
    return total_profit,

# 解码个体
def decode_individual(individual, num_years, land_areas, valid_crops):
    decoded_individual = {}
    index = 0
    for year in range(num_years):
        for land_type, plots in land_areas.items():
            for plot in plots:
                for crop in valid_crops:
                    # 假设每个基因代表作物在地块上的种植比例
                    decoded_individual[(land_type, plot, crop, year)] = individual[index] / len(valid_crops) * land_areas[land_type][plot]
                    index += 1
    return decoded_individual

# 判断地块名称是否连续
def is_consecutive(plots):
    plots_sorted = sorted([int(plot[1:]) for plot in plots])
    return all(plots_sorted[i] + 1 == plots_sorted[i + 1] for i in range(len(plots_sorted) - 1))


# 交叉操作
toolbox.register("mate", tools.cxTwoPoint)
# 变异操作
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# 选择操作
toolbox.register("select", tools.selTournament, tournsize=3)
# 注册评估函数
toolbox.register("evaluate", evaluate, scenarios=scenarios, land_areas=land_areas, valid_crops=valid_crops, actual_planting_2023=actual_planting_2023)

# 初始化种群
POPULATION_SIZE = 100
HALL_OF_FAME_SIZE = 1
population = [toolbox.individual() for _ in range(POPULATION_SIZE)]
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# 进化过程
NGEN = 250
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.005, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

# 输出最优解
best_individual = hof[0]
decoded_best_individual = decode_individual(best_individual, num_years, land_areas, valid_crops)

print("Best Individual Fitness:", best_individual.fitness.values)
print("Decoded Best Individual:")

# 提取最优解中的具体种植面积信息
results = []
for year in range(len(scenarios)):
    for land_type, plots in land_areas.items():
        for plot in plots:
            for crop in valid_crops:
                # 获取特定年份、土地类型、地块和作物的种植面积
                planting_area = decoded_best_individual.get((land_type, plot, crop, year), 0)
                if planting_area > 0:
                    results.append({
                        '年份': 2024 + year,
                        '地块类型': land_type,
                        '地块名称': plot,
                        '作物名称': crop,
                        '种植面积': planting_area
                    })
                    
# 保存结果到Excel
output_path = r'D:\数模\C题\第二问草稿\单季\种植方案.xlsx'
output_df = pd.DataFrame(results)
with pd.ExcelWriter(output_path) as writer:
    output_df.to_excel(writer, sheet_name='种植方案', index=False)

print(f"Results saved to {output_path}")