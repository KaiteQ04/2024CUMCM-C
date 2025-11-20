import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms

# 读取Excel文件
sales_results_path = r'D:\数模\C题\第二问草稿\双季\使用数据\连接结果（双季）.xlsx'
crop_types_path = r'D:\数模\C题\第二问草稿\双季\使用数据\每种作物适合种植的地块类型与季别_合并.xlsx'
crop_varieties_path = r'D:\数模\C题\第二问草稿\双季\使用数据\作物种植限制.xlsx'

sales_results_df = pd.read_excel(sales_results_path, sheet_name='2023销售结果')
analysis_df = pd.read_excel(sales_results_path, sheet_name='销售结果分析')
crop_types_df = pd.read_excel(crop_types_path)
crop_varieties_df = pd.read_excel(crop_varieties_path)

# 生成 valid_crops 字典，存储有效的作物及其是否为豆类
valid_crops_s1 = {}
valid_crops_s2 = {}
valid_crops_p1 = {}
valid_crops_p2 = {}
valid_crops_z1 = {}
valid_crops_z2 = {}

for index, row in crop_varieties_df.iterrows():
    if row['地块类型'] == '水浇地第一季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_s1[crop_name] = is_legume
    elif row['地块类型'] == '水浇地第二季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_s2[crop_name] = is_legume
        
for index, row in crop_varieties_df.iterrows():
    if row['地块类型'] == '普通大棚第一季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_p1[crop_name] = is_legume
    elif row['地块类型'] == '普通大棚第二季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_p2[crop_name] = is_legume
        
for index, row in crop_varieties_df.iterrows():
    if row['地块类型'] == '智慧大棚第一季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_z1[crop_name] = is_legume
    elif row['地块类型'] == '智慧大棚第二季':
        crop_name = row['作物名称']
        crop_type = row['作物类型']
        is_legume = '豆类' in crop_type
        valid_crops_z2[crop_name] = is_legume

# 2023年的实际种植数据
actual_planting_2023 = {}
fields_to_keep = ['作物名称', '作物类型', '地块名称', '亩产量/斤', '地块类型', '种植成本/(元/斤)', '种植面积/亩', '作物产量（销量）', '实际销售价格']

for index, row in sales_results_df.iterrows():
    plot_name = row['地块名称']
    crop_name = row['作物名称']
    crop_info = sales_results_df[sales_results_df['作物名称'] == crop_name]
    if not crop_info.empty:
        crop_type = crop_info.iloc[0]['作物类型']
        is_legume = '豆类' in crop_type
        for field in fields_to_keep:
            value = row[field] if field != '作物类型' else crop_type
            setattr(row, f'field_{field}', value)
        actual_planting_2023[(plot_name, crop_name)] = {f'field_{field}': getattr(row, f'field_{field}') for field in fields_to_keep}

# 地块面积信息
land_areas = {
    '水浇地第一季': {'D1': 15, 'D2': 10, 'D3': 14, 'D4': 6, 'D5': 10, 'D6': 12, 'D7': 22, 'D8': 20},
    '普通大棚第一季': {'E1': 0.6, 'E2': 0.6, 'E3': 0.6, 'E4': 0.6, 'E5': 0.6, 'E6': 0.6, 'E7': 0.6, 'E8': 0.6,
                     'E9': 0.6, 'E10': 0.6, 'E11': 0.6, 'E12': 0.6, 'E13': 0.6, 'E14': 0.6, 'E15': 0.6, 'E16': 0.6},
    '智慧大棚第一季': {'F1': 0.6, 'F2': 0.6, 'F3': 0.6, 'F4': 0.6},
    '水浇地第二季': {'D1': 15, 'D2': 10, 'D3': 14, 'D4': 6, 'D5': 10, 'D6': 12, 'D7': 22, 'D8': 20},
    '普通大棚第二季': {'E1': 0.6, 'E2': 0.6, 'E3': 0.6, 'E4': 0.6, 'E5': 0.6, 'E6': 0.6, 'E7': 0.6, 'E8': 0.6,
                     'E9': 0.6, 'E10': 0.6, 'E11': 0.6, 'E12': 0.6, 'E13': 0.6, 'E14': 0.6, 'E15': 0.6, 'E16': 0.6},
    '智慧大棚第二季': {'F1': 0.6, 'F2': 0.6, 'F3': 0.6, 'F4': 0.6}
}

# 创建DEAP的creator对象
try:
    del creator.FitnessMax
except AttributeError:
    pass

try:
    del creator.Individual
except AttributeError:
    pass

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 创建toolbox
toolbox = base.Toolbox()

# 注册生成器
toolbox.register("attr_bool", random.randint, 0, 1)

# 计算个体长度
num_plots = sum([len(plots) for plots in land_areas.values()])
num_years = 7  # 2024-2030
individual_length = num_plots * num_years * len(valid_crops_s1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=individual_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 蒙特卡洛模拟
def monte_carlo_simulation(actual_data, years):
    simulated_data = []
    for year in range(years):
        temp_data = {}
        for (plot, crop), data in actual_data.items():
            # 销售量
            if crop in ['小麦', '玉米']:
                sale = data['field_作物产量（销量）'] * (1 + np.random.uniform(0.05, 0.10) * year)
            else:
                sale = data['field_作物产量（销量）'] * (1 + np.random.uniform(-0.05, 0.05))
            
            # 亩产量
            yield_per_acre = data['field_亩产量/斤'] * (1 + np.random.uniform(-0.10, 0.10))
            
            # 种植成本
            cost_per_acre = data['field_种植成本/(元/斤)'] * (1 + 0.05 * year)
            
            # 销售价格
            if data['field_作物类型'] == '粮食类':
                selling_price = data['field_实际销售价格'] * (1 + 0.01 * year)
            elif data['field_作物类型'] == '蔬菜类':
                selling_price = data['field_实际销售价格'] * (1 + 0.05 * year)
            elif data['field_作物类型'] == '食用菌':
                if crop == '羊肚菌':
                    selling_price = data['field_实际销售价格'] * (1 - 0.05 * year)
                else:
                    selling_price = data['field_实际销售价格'] * (1 + np.random.uniform(-0.01, -0.05))
            else:
                # 处理未分类的作物类型
                selling_price = data['field_实际销售价格'] * (1 + np.random.uniform(-0.05, 0.05) * year)
            
            temp_data[(plot, crop)] = {
                'yield_per_acre': yield_per_acre,
                'selling_price': selling_price,
                'cost_per_acre': cost_per_acre,
                'sale': sale
            }
        simulated_data.append(temp_data)
    return simulated_data

# 生成多种情景
def generate_scenarios(simulated_data, years):
    scenarios = []
    for year in years:
        temp_data = {}
        for (plot, crop), values in simulated_data[year - 2024].items():
            # 预期销售量
            if crop in ['小麦', '玉米']:
                temp_data[(plot, crop)] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(0.05, 0.10)),
                    'selling_price': values['selling_price'] * (1 + np.random.uniform(0.01, 0.01)),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(0.05, 0.05)),
                    'sale': values['sale']
                }
            elif crop == '羊肚菌':
                temp_data[(plot, crop)] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(-0.10, 0.10)),
                    'selling_price': values['selling_price'] * (1 - np.random.uniform(0.05, 0.05)),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(0.05, 0.05)),
                    'sale': values['sale']
                }
            else:
                temp_data[(plot, crop)] = {
                    'yield_per_acre': values['yield_per_acre'] * (1 + np.random.uniform(-0.10, 0.10)),
                    'selling_price': values['selling_price'] * (1 + np.random.uniform(-0.01, -0.05)),
                    'cost_per_acre': values['cost_per_acre'] * (1 + np.random.uniform(0.05, 0.05)),
                    'sale': values['sale']
                }
        scenarios.append(temp_data)
    return scenarios

# 使用蒙特卡洛模拟生成未来7年的数据
simulated_data = monte_carlo_simulation(actual_planting_2023, 7)

# 生成多种情景
years = range(2024, 2031)
scenarios = generate_scenarios(simulated_data, years)

# 适应度函数
def evaluate(individual, scenarios, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2, actual_planting_2023):
    total_profit = 0
    decoded_individual = decode_individual(individual, num_plots, num_years, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2)
    
    # 检查约束条件
    if not check_constraints(decoded_individual, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2, scenarios):
        return -1,
    
    # 计算总利润
    for year in range(len(scenarios)):
        for land_type, plots in land_areas.items():
            valid_crops_for_type = get_valid_crops(land_type, year)
            for plot, area in plots.items():
                for crop in valid_crops_for_type:
                    if decoded_individual.get((land_type, plot, crop, year), 0) > 0:
                        profit = (scenarios[year][(plot, crop)]['yield_per_acre'] * scenarios[year][(plot, crop)]['selling_price'] - scenarios[year][(plot, crop)]['cost_per_acre']*scenarios[year][(plot, crop)]['yield_per_acre']) * decoded_individual.get((land_type, plot, crop, year), 0)
                        total_profit += profit
    return total_profit,

def decode_individual(individual, num_plots, num_years, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2):
    decoded_individual = {}
    plot_counter = 0  # 用于追踪地块的计数器

    # 遍历所有地块
    for land_type, plots in land_areas.items():
        for plot, area in plots.items():
            for year in range(num_years):
                valid_crops = get_valid_crops(land_type, year)
                
                # 对于每一种有效作物，检查个体中是否有种植标记
                for j, crop in enumerate(valid_crops):
                    key = (land_type, plot, crop, year)
                    if key not in decoded_individual:
                        decoded_individual[key] = 0
                    
                    # 通过索引找到对应的基因位
                    gene_index = plot_counter * len(valid_crops) + j
                    if individual[gene_index]:
                        decoded_individual[key] = 1
                
                # 更新地块计数器
                plot_counter += 1
            
            # 每个地块类型中的地块都处理完后，重置计数器
            if plot_counter >= num_plots:
                plot_counter = 0

    return decoded_individual

# 获取对应地块类型的作物
def get_valid_crops(land_type, year):
    if '水浇地第一季' in land_type:
        return valid_crops_s1
    elif '水浇地第二季' in land_type:
        return valid_crops_s2
    elif '普通大棚第一季' in land_type:
        return valid_crops_p1
    elif '普通大棚第二季' in land_type:
        return valid_crops_p2
    elif '智慧大棚第一季' in land_type:
        return valid_crops_z1
    elif '智慧大棚第二季' in land_type:
        return valid_crops_z2
    return {}

# 检查约束条件
def check_constraints(decoded_individual, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2, scenarios):
    # 耕地面积限制
    for year in range(len(scenarios)):
        plot_total_areas = {}
        for land_type, plots in land_areas.items():
            valid_crops_for_type = get_valid_crops(land_type, year)
            for plot, area in plots.items():
                plot_total_areas.setdefault(plot, 0)
                for crop in valid_crops_for_type:
                    plot_total_areas[plot] += decoded_individual.get((land_type, plot, crop, year), 0)
                if plot_total_areas[plot] > area:
                    return False

    # 不重茬约束
    for land_type, plots in land_areas.items():
        for plot in plots:
            crops = [None] * len(scenarios)
            for year in range(len(scenarios)):
                valid_crops_for_type = get_valid_crops(land_type, year)
                for crop in valid_crops_for_type:
                    if decoded_individual.get((land_type, plot, crop, year), 0) > 0:
                        crops[year] = crop
                        break
            for year in range(len(scenarios) - 1):
                if crops[year] == crops[year + 1]:
                    return False

    # 豆类作物三年内种植一次的约束
    for land_type, plots in land_areas.items():
        valid_crops_for_type = get_valid_crops(land_type, year)
        for plot in plots:
            legume_years = [year for year in range(len(scenarios)) if any(valid_crops_for_type[crop] for crop in valid_crops_for_type if decoded_individual.get((land_type, plot, crop, year), 0) > 0)]
            if legume_years and max(legume_years) - min(legume_years) > 3:
                return False

    # 每种作物每季的种植地不能太分散
    for crop in valid_crops_s1.keys():  # 假设所有作物都在第一季种植
        for year in range(len(scenarios)):
            plots_per_type = {}
            for land_type, plots in land_areas.items():
                valid_crops_for_type = get_valid_crops(land_type, year)
                if crop in valid_crops_for_type:
                    planted_plots = [plot for plot in plots if decoded_individual.get((land_type, plot, crop, year), 0) > 0]
                    if planted_plots:
                        plots_per_type[land_type] = planted_plots
            for land_type, planted_plots in plots_per_type.items():
                if len(planted_plots) > 1:
                    if not is_consecutive(planted_plots):
                        return False
    return True

# 判断地块名称是否连续
def is_consecutive(plots):
    plots_sorted = sorted([int(plot[1:]) for plot in plots])  # 假设地块名为 D1, D2, D3 ...
    return all(plots_sorted[i] + 1 == plots_sorted[i + 1] for i in range(len(plots_sorted) - 1))

# 注册evaluate函数
toolbox.register("evaluate", evaluate, scenarios=scenarios, land_areas=land_areas, valid_crops_s1=valid_crops_s1, valid_crops_s2=valid_crops_s2, valid_crops_p1=valid_crops_p1, valid_crops_p2=valid_crops_p2, valid_crops_z1=valid_crops_z1, valid_crops_z2=valid_crops_z2, actual_planting_2023=actual_planting_2023)

# 注册其他操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 初始化种群
POPULATION_SIZE = 100
HALL_OF_FAME_SIZE = 1
population = toolbox.population(n=POPULATION_SIZE)
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
decoded_best_individual = decode_individual(best_individual, num_plots, num_years, land_areas, valid_crops_s1, valid_crops_s2, valid_crops_p1, valid_crops_p2, valid_crops_z1, valid_crops_z2)

print("Best Individual Fitness:", best_individual.fitness.values)
print("Decoded Best Individual:")

# 提取最优解中的具体种植面积信息
results = []
for year in range(len(scenarios)):
    for land_type, plots in land_areas.items():
        for plot in plots:
            for crop in get_valid_crops(land_type, year):
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
output_path = r'D:\数模\C题\第二问草稿\双季\种植方案.xlsx'
output_df = pd.DataFrame(results)
with pd.ExcelWriter(output_path) as writer:
    output_df.to_excel(writer, sheet_name='种植方案', index=False)

print(f"Results saved to {output_path}")