import random

# --- 1. 定义常量 ---
PROB_RED = 0.004        # 单个盒子抽出红球的概率
NUM_TYPES = 21          # 红球种类 (A1 到 A21)
NUM_BOXES = 4           # 盒子数量 (孔数)
NUM_SIMULATIONS = 10000    # 我们要模拟i次 "找到A1" 的过程

# --- 2. 计算成功的概率 ---

# 假设21种红球是等概率出现的
# P(A1) = P(抽中红球) * P(红球是A1 | 抽中红球)
# 在单个盒子中，抽中指定红球 "A1" 的概率
PROB_A1_ONE_BOX = PROB_RED / NUM_TYPES

# P(单次淬炼失败) = P(4个盒子 *都* 没有抽中A1)
# P(单个盒子没有抽中A1) = 1 - PROB_A1_ONE_BOX
prob_fail_one_box = 1 - PROB_A1_ONE_BOX
prob_fail_all_four_boxes = prob_fail_one_box ** NUM_BOXES

# P(单次淬炼成功) = P(4个盒子中 *至少有1个* 抽中A1)
# P(成功) = 1 - P(4个盒子都失败)
PROB_SUCCESS_PER_TRY = 1 - prob_fail_all_four_boxes

# 理论期望次数 (用于对比)
# 这是一个几何分布，期望 E = 1 / P(成功)
THEORETICAL_EXPECTATION = 1 / PROB_SUCCESS_PER_TRY

# --- 3. 定义模拟函数 ---

def simulate_one_run(prob_success):
    """
    模拟一次实验，直到成功为止。
    返回所需的尝试次数。
    """
    tries_count = 0
    while True:
        # 进行了第N次尝试
        tries_count += 1 
        
        # 模拟这次尝试(4个孔)是否成功
        # random.random() 会生成一个 [0.0, 1.0) 之间的随机数
        if random.random() < prob_success:
            return tries_count # 成功了，返回尝试次数

# --- 4. 执行模拟 ---
print("====== 开始模拟抽取指定红球 (A1) ======")
print(f"系统设置: 4个盒子, 21种红球, 红色概率 0.4%")
print(f"单次淬炼 (4孔) 成功的精确概率: {PROB_SUCCESS_PER_TRY:.8f} (或 1 / {THEORETICAL_EXPECTATION:.2f})")
print(f"理论平均期望: {THEORETICAL_EXPECTATION:.2f} 次")
print("------------------------------------------")
print(f"开始模拟 {NUM_SIMULATIONS} 次，记录每次成功所需的抽取次数:")

results = []
for i in range(NUM_SIMULATIONS):
    # 调用函数，看这一次花了多少次
    tries_needed = simulate_one_run(PROB_SUCCESS_PER_TRY)
    results.append(tries_needed)
    # 使用 :2d 和 :5d 来格式化输出，使其对齐
    print(f"  第 {i+1:2d} 次模拟: 成功! 用了 {tries_needed:5d} 次")

# --- 5. 结果汇总 ---
print("------------------------------------------")
print("模拟结束。")
#print(f"模拟的结果 (每次所需的次数): {results}")
print(max(results))
# 计算10次模拟的平均值
average_tries = sum(results) / len(results)
print(f"模拟的平均次数: {average_tries:.2f} 次")
print(f"(这与理论期望的 {THEORETICAL_EXPECTATION:.2f} 次接近)")