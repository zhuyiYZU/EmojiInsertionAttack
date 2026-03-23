# ================================
# 手动填写下面这两个数值
# ================================
orig_correct = 0      # 攻击前预测正确的样本数
total_samples = 0     # 总样本数 len(texts)
attack_changed = 0    # 原本预测正确的样本中，被攻击后改判的样本数

# ================================
# 指标计算
# ================================

# 1. 初始准确率
if total_samples == 0:
    init_acc = 0
else:
    init_acc = orig_correct / total_samples * 100

# 2. 攻击成功率
if orig_correct == 0:
    attack_success_rate = 0
else:
    attack_success_rate = attack_changed / orig_correct * 100

# 3. 攻击后的准确率
if total_samples == 0:
    attack_acc = 0
else:
    attack_acc = orig_correct * (1 - attack_success_rate / 100) / total_samples * 100

# ================================
# 输出结果
# ================================
print(f"初始准确率 init_acc: {init_acc:.2f}%")
print(f"攻击成功率 attack_success_rate: {attack_success_rate:.2f}%")
print(f"攻击后的准确率 attack_acc: {attack_acc:.2f}%")