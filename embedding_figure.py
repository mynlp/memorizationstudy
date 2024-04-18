import numpy as np
import matplotlib.pyplot as plt


num_points = 16
r = np.random.uniform(5, 8, num_points)
theta_peak = np.pi / 2  # 偏好角度峰值
# 产生权重，用以生成偏向特定角度区间的随机值
weights = np.abs(np.random.normal(loc=theta_peak, scale=np.pi / 10, size=num_points)) % (2 * np.pi)
theta = weights
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.figure(figsize=(12, 8))

# 画起始点，使用更大的点和明显的颜色标记起点
plt.scatter(x, y, s=120, color='dodgerblue', edgecolor='black', linewidth=1, label='Starting Points', zorder=5)

for xi, yi in zip(x, y):
    dir_to_origin = np.array([-xi, -yi])
    dir_to_origin /= np.linalg.norm(dir_to_origin)
    perp_dir = np.array([-dir_to_origin[1], dir_to_origin[0]])

    start_x, start_y = xi, yi
    steps = 4
    for step in range(steps):
        offset = perp_dir * 0.2 if step % 2 == 0 else -perp_dir * 0.2
        xi, yi = xi + dir_to_origin[0] * 1 / steps + offset[0], yi + dir_to_origin[1] * 1 / steps + offset[1]

        if step == 0:  # 使起始锯齿线更明显
            plt.plot([start_x, xi], [start_y, yi], '-o', color='skyblue', markersize=4, linewidth=2)
        else:
            plt.plot([start_x, xi], [start_y, yi], '-', color='skyblue', linewidth=2)
        start_x, start_y = xi, yi

    # 缩短直线长度，并减少箭头大小
    end_x, end_y = xi + dir_to_origin[0] * 0.1, yi + dir_to_origin[1] * 0.1
    plt.plot([start_x, end_x], [start_y, end_y], '-', color='limegreen', linewidth=2.5, alpha=0.7)
    plt.arrow(end_x, end_y, dir_to_origin[0] * 0.1, dir_to_origin[1] * 0.1, fc='limegreen', ec='limegreen',
              head_width=0.2, head_length=0.3, alpha=0.7)

# 标记原点
plt.scatter(0, 0, s=200, color='red', marker='.', label='Origin', zorder=10)

plt.title('Points Moving Toward Origin with Simplified Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.show()