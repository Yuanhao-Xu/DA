# CreatTime 2024/7/18
import matplotlib.pyplot as plt

# 导入数据
from RS_main import RS_R2_Score
from LL_main import LL_R2_Score


# Active loop counts
x = range(1, len(RS_R2_Score) + 1)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x, RS_R2_Score, marker='o', label='NN_RS')
plt.plot(x, LL_R2_Score, marker='s', label='LL4AL')
plt.ylim(0.3, 1)
plt.xlim(0, len(RS_R2_Score))

# Adding titles and labels
plt.title('Active Learning Accuracy Over Cycles')
plt.xlabel('Active Learning Cycle')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show(block=True)




print(f"RS_R2_Score:{RS_R2_Score}")
print(f"LL_R2_Score:{LL_R2_Score}")