# 先定义时间间隔（以秒为单位）
dt = 0.005  # 这个值应根据你的实际情况进行调整

# 打开文件并按行读取速度数据
with open('/home/jjq/motion_imitation/mpc_controller/data/lower_velocity.txt', 'r') as file:
    lines = file.readlines()

# 将字符串转换为浮点数
speeds = [float(line.strip()) for line in lines]

# 计算加速度
accelerations = [(speeds[i+1] - speeds[i]) / dt for i in range(len(speeds) - 1)]

print(len(accelerations))

print(accelerations[0:99])
