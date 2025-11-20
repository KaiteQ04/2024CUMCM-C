import pandas as pd
from deap import base, creator, tools, algorithms
import numpy as np
from random import randint, choice
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os

# 数据路径
planting_restrictions_path = r'D:\数模\C题\第一题\第二问\双季作物\使用数据\作物种植限制.xlsx'
sales_results_path = r'D:\数模\C题\数据预处理\第一题第一问\连接结果（双季）.xlsx'
suitability_path = r'D:\数模\C题\数据预处理\第一题第一问\每种作物适合种植的地块类型与季别_合并.xlsx'

# 加载数据
planting_restrictions_df = pd.read_excel(planting_restrictions_path)
sales_results_df = pd.read_excel(sales_results_path, sheet_name='2023销售结果')

# 地块信息
land_info = {
    "水浇地第一季": {"areas": [15, 10, 14, 6, 10, 12, 22, 20], "names": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]},
    "水浇地第二季": {"areas": [15, 10, 14, 6, 10, 12, 22, 20], "names": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]},
    "普通大棚第一季": {"areas": [0.6] * 16, "names": ["E{}".format(i+1) for i in range(16)]},
    "普通大棚第二季": {"areas": [0.6] * 16, "names": ["E{}".format(i+1) for i in range(16)]},
    "智慧大棚第一季": {"areas": [0.6] * 4, "names": ["F{}".format(i+1) for i in range(4)]},
    "智慧大棚第二季": {"areas": [0.6] * 4, "names": ["F{}".format(i+1) for i in range(4)]}
}

# 获取每种地块类型允许种植的作物列表
crop_info = {}
for plot_type in land_info.keys():
    allowed_crops = planting_restrictions_df[planting_restrictions_df['地块类型'] == plot_type][['作物名称', '作物类型']]
    crop_info[plot_type] = allowed_crops.to_dict(orient='records')

# 销售信息
sales_info = {}
for _, row in sales_results_df.iterrows():
    sales_info[row['作物名称']] = {'expected_sales': row['作物产量（销量）'], 'price': row['实际销售价格'], 'cost': row['种植成本/(元/亩)'], 'discount_price': row['实际销售价格'] * 0.5}

# 初始化DEAP框架
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", lambda: randint(0, len(crop_info[list(crop_info.keys())[0]])-1) if len(crop_info[list(crop_info.keys())[0]]) > 0 else -1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, len(land_info.keys()) * max(len(v['areas']) for v in land_info.values()))

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 适应度函数
def evaluate(individual):
    fitness = 0
    plot_usage = {plot: 0 for plot in land_info.keys()}
    plot_crops = {plot: [] for plot in land_info.keys()}
    crop_production = {crop['作物名称']: 0 for plot in land_info.keys() for crop in crop_info[plot]}
    plots_used = {plot: False for plot in land_info.keys()}
    
    index = 0
    for plot, info in land_info.items():
        areas = info['areas']
        names = info['names']
        plot_type = plot
        for i, name in enumerate(names):
            crop_index = individual[index]
            index += 1
            if crop_index >= 0 and crop_index < len(crop_info[plot_type]):
                crop = crop_info[plot_type][crop_index]
                crop_name = crop['作物名称']
                plot_area = areas[i % len(areas)]
                
                # 检查地块的使用面积
                if plot_usage[plot] + plot_area > sum(areas):
                    return -1,
                
                plot_usage[plot] += plot_area
                plot_crops[plot].append((crop_name, plot_area))
                crop_production[crop_name] += plot_area
                
                # 检查作物是否有销售信息
                if crop_name not in sales_info:
                    continue
                
                if crop_production[crop_name] > sales_info[crop_name]['expected_sales']:
                    regular_sales = sales_info[crop_name]['expected_sales']
                    excess_sales = crop_production[crop_name] - regular_sales
                    crop_production[crop_name] = regular_sales + excess_sales
                    
                    # 计算利润
                    regular_profit = regular_sales * (sales_info[crop_name]['price'] - sales_info[crop_name]['cost'])
                    excess_profit = excess_sales * ((sales_info[crop_name]['price'] * 0.5) - sales_info[crop_name]['cost'])
                    fitness += regular_profit + excess_profit
                else:
                    # 计算利润
                    profit = crop_production[crop_name] * (sales_info[crop_name]['price'] - sales_info[crop_name]['cost'])
                    fitness += profit
                plots_used[plot] = True
    
    # 检查豆类作物种植
    for plot in plots_used:
        if not plots_used[plot]:
            continue
        
        crops_in_plot = [crop for crop, _ in plot_crops[plot]]
        if not any('豆类' in crop['作物类型'] for crop in crop_info[plot]):
            return -1,
    
    # 检查连续重茬种植
    for plot in plots_used:
        crops_in_plot = [crop for crop, _ in plot_crops[plot]]
        if len(set(crops_in_plot)) < len(crops_in_plot):
            return -1,
    
    return fitness,

# 交叉操作
toolbox.register("mate", tools.cxTwoPoint)
# 变异操作
toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(len(v) for v in crop_info.values()), indpb=0.05)
# 选择操作
toolbox.register("select", tools.selTournament, tournsize=3)
# 评价操作
toolbox.register("evaluate", evaluate)

# 主函数
def main():
    years = range(2024, 2031)
    
    for year in years:
        # 初始化种群
        pop = toolbox.population(n=50)
        
        # 进化过程
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, verbose=True)
        
        # 找出最佳个体
        best_ind = tools.selBest(pop, 1)[0]
        
        # 将结果保存到Excel
        save_results_to_excel(best_ind, year)

def save_results_to_excel(individual, year):
    results = []
    index = 0
    for plot, info in land_info.items():
        areas = info['areas']
        names = info['names']
        for i, name in enumerate(names):
            crop_index = individual[index]
            index += 1
            if crop_index >= 0 and crop_index < len(crop_info[plot]):
                crop = crop_info[plot][crop_index]
                plot_area = areas[i % len(areas)]
                results.append({
                    '年份': year,
                    '地块类型': plot,
                    '地块名称': name,
                    '种植的作物名称': crop['作物名称'],
                    '种植作物的面积': plot_area
                })
    
    # 创建Excel文件
    filename = f'results_{year}.xlsx'
    if os.path.exists(filename):
        mode = 'a'  # Append mode if the file exists
    else:
        mode = 'w'  # Write mode if the file does not exist
    
    wb = Workbook()
    ws = wb.active
    ws.append(['年份', '地块类型', '地块名称', '种植的作物名称', '种植作物的面积'])
    
    for result in results:
        ws.append([
            result['年份'],
            result['地块类型'],
            result['地块名称'],
            result['种植的作物名称'],
            result['种植作物的面积']
        ])
    
    try:
        wb.save(filename)
    except PermissionError:
        raise PermissionError(f"Could not write to '{filename}'. Please close any open instances of this file.")

if __name__ == "__main__":
    main()