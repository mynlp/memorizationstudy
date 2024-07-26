import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap

def arrows_draw(x, y):
    for xi, yi in zip(x, y):
        dir_to_origin = np.array([-xi, -yi])
        dir_to_origin /= np.linalg.norm(dir_to_origin)
        perp_dir = np.array([-dir_to_origin[1], dir_to_origin[0]])
        start_x, start_y = xi, yi
        steps = 3
        for step in range(steps):
            offset = perp_dir * 0.1 if step % 2 == 0 else -perp_dir * 0.2
            xi, yi = xi + dir_to_origin[0] * 0.5 / steps + offset[0], yi + dir_to_origin[1] * 0.5 / steps + offset[1]

            if step == 0:  # 使起始锯齿线更明显
                plt.plot([start_x, xi], [start_y, yi], '-o', color='skyblue', markersize=4, linewidth=2)
            else:
                plt.plot([start_x, xi], [start_y, yi], '-', color='skyblue', linewidth=2)
            start_x, start_y = xi, yi

        # 缩短直线长度，并减少箭头大小
        end_x, end_y = xi + dir_to_origin[0] * 0.03, yi + dir_to_origin[1] * 0.03
        plt.plot([start_x, end_x], [start_y, end_y], '-', color='limegreen', linewidth=2.5, alpha=0.7)
        plt.arrow(end_x, end_y, dir_to_origin[0] * 0.03, dir_to_origin[1] * 0.03, fc='limegreen', ec='limegreen',
                  head_width=0.2, head_length=0.3, alpha=0.7)

def label_points(x, y, legend_added):
    indices = np.argsort(x)  # 对x坐标进行排序以决定标签的顺序
    cmap = get_cmap('viridis', num_points)  # 使用matplotlib的颜色映射
    colors = [cmap(i) for i in range(num_points)]
    for i, idx in enumerate(indices):
        label = f' {i} Token Memorized'
        if label not in legend_added:
            plt.scatter(x[idx], y[idx], color=colors[i], s=120, edgecolor='black', label=label, zorder=5)
            legend_added.add(label)
        else:
            plt.scatter(x[idx], y[idx], color=colors[i], s=120, edgecolor='black', zorder=5)
    return colors, legend_added

def plot_annular_sector(x, y, label, color):
    radii = np.sqrt(x ** 2 + y ** 2)
    max_radius_with_arrow = max(radii)   # 增加一些长度以包括箭头
    min_radius_with_arrow = max(0, min(radii) - 1)  # 减少半径以考虑箭头的起始位置

    angles = np.arctan2(y, x)
    min_angle = min(angles)
    max_angle = max(angles)

    width = max_radius_with_arrow - min_radius_with_arrow
    sector = patches.Wedge((0, 0), max_radius_with_arrow, np.degrees(min_angle), np.degrees(max_angle), width=width,
                           edgecolor='black', facecolor=color, alpha=0.15)
    plt.gca().add_patch(sector)
    offset = 0.5  # adjust this value according to your graph's scale
    text_radius = min_radius_with_arrow + offset
    text_angle = min_angle  # position the text at left edge of the sector
    plt.text(text_radius * np.cos(text_angle)+1, text_radius * np.sin(text_angle), f"{label}",
             color="black", fontsize=12, ha='center', rotation=np.degrees(text_angle))
    # mid_radius = (min_radius_with_arrow + max_radius_with_arrow) / 2
    # mid_angle = (min_angle + max_angle) / 2
    # plt.text(mid_radius * np.cos(mid_angle), mid_radius * np.sin(mid_angle), f"{label}",
    #          color="black", fontsize=12, ha='center')



num_points = 16
theta_peak = np.pi / 4  # 偏好角度峰值：90度
concentration = np.pi / 6  # 分布集中在45度左右

# 基本角度间隔
base_increment1 = concentration / (num_points / 4.0)  # 内层较密集分布
base_increment2 = concentration / (num_points / 3.0)  # 外层较稀疏
base_increment3 = concentration / (num_points / 3.0)
# 添加随机扰动，保证随机性同时避免太大的重叠
angle_jitters1 = np.random.uniform(-base_increment1/2, base_increment1/2, num_points)
angle_jitters2 = np.random.uniform(-base_increment2/2, base_increment2/2, num_points)
angle_jitters3 = np.random.uniform(-base_increment3/2, base_increment3/2, num_points)

# 生成角度：基于theta_peak + 角度增量的累加
theta1 = np.cumsum(np.full(num_points, base_increment1) + angle_jitters1) + theta_peak# - concentration/2
theta2 = np.cumsum(np.full(num_points, base_increment2) + angle_jitters2) + theta_peak# - concentration/2
theta3 = np.cumsum(np.full(num_points, base_increment3) + angle_jitters3) + theta_peak# - concentration/2

# 生成半径
r1 = np.random.uniform(2, 4, num_points)
r2 = np.random.uniform(6, 8, num_points)
r3 = np.random.uniform(10, 12, num_points)

# 将极坐标转化为笛卡尔坐标
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1)
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
x3 = r3 * np.cos(theta3)
y3 = r3 * np.sin(theta3)




plt.figure(figsize=(10, 8))

# 画起始点，使用更大的点和明显的颜色标记起点
plt.scatter(x1, y1, s=120, color='dodgerblue', edgecolor='black', linewidth=1, zorder=5)
plt.scatter(x2, y2, s=120, color='yellow', edgecolor='black', linewidth=1, zorder=5)
plt.scatter(x3, y3, s=120, color='red', edgecolor='black', linewidth=1, zorder=5)

legend = set()
colors1, legend = label_points(x1, y1, legend)
colors2, legend = label_points(x2, y2, legend)
colors3, legend = label_points(x3, y3, legend)
arrows_draw(x1, y1)
arrows_draw(x2, y2)
arrows_draw(x3, y3)

plot_annular_sector(x1, y1, '410m' , 'black')
plot_annular_sector(x2, y2, '2.8b', 'black')
plot_annular_sector(x3, y3, '12b', 'black')
# 标记原点
plt.scatter(0, 0, s=200, color='black', marker='.', zorder=10)
# plt.axvline(x=0, color='black', linewidth=1)  # 垂直线表示X轴
# plt.axhline(y=0, color='black', linewidth=1)  # 水平线表示Y轴
# arrow = patches.FancyArrowPatch(
#     (x3[-1], y3[-1]+2),
#     (x3[0], y3[0]+2),
#     connectionstyle="arc3,rad=-.4",
#     color="black",
#     arrowstyle="<|-",
#     mutation_scale=20,
#     linewidth=2)

plt.gca().annotate(
    '',  # No annotation text
    xy=(x3[0]+1, y3[0] + 1.5),  # The point that the arrow points to
    xytext=(x3[-1]-1, y3[-1] + 1.5),  # The starting point of arrow
    textcoords='data',
    arrowprops=dict(
        arrowstyle="<|-",
        connectionstyle="arc3,rad=-.3",
        color="black",
        mutation_scale=20,
        linewidth=2)
)

# Text at the middle of arrow
plt.gca().text(
    (x3[0] + x3[-1]) / 2,  # x-coordinate of text (middle point of arrow)
    (y3[0] + y3[-1]) / 2 + 5,  # y-coordinate of text (middle point of arrow)
    'Memorization Score Decreasing',  # Your annotation text
    verticalalignment='center', horizontalalignment='center',  # Centered text
    fontsize=12,  # font size of the text
    color='black'  # font color of the text
)
#plt.gca().add_patch(arrow)
plt.title('Embedding Dynamics of Sentences with Different Memorization Scores', fontsize=16)
plt.xlabel('X Axis', fontsize=14)
plt.ylabel('Y Axis', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')
plt.xlim(-12, 12)
y_max = max(np.max(y1), np.max(y2), np.max(y3))  # 确保包含所有组的最大值
plt.ylim(-0.5, y_max + 0.5)
plt.savefig('plot.png', bbox_inches='tight', dpi=600)
plt.show()