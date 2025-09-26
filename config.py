# 带宽配置
# # v2i_band = 8
# v2i_band = 5
# # core_band = 15
# # ceil_band = 3# 3
# ceil_band = 2
# uav之间通过通信带宽 原始是7
core_band = 10

u2c_band = 8

g2u_band = 6

g2h_band = 2

base_pos = ((150, 150), (150, 450), (150, 750), (450, 150), (450, 450), (450, 750), (750, 150), (750, 450), (750, 750))
base_cover = 200
# 无人机高度和高空平台高度
# uav_h = 400
# hap_h = 1000
# 地面设备位置
ground_pos = ((150,450),(450, 150), (450, 450), (450, 750),(150,750),(750,750))

MEMORY_CAPACITY = 300
MEMORY_CAPACITY_AC = 300

f_range_mfn = (7, 10)
# c_range_mfn = (30, 50)
c_range_mfn = (50, 80)

# f_range_sfn = (30,50)
# c_range_sfn = (80, 100)
# f_sfn =(35, 30, 35, 32)     # [12, 16]均匀分布随机生成62350
f_uav =(80, 84, 88, 90, 94, 96, 100, 93, 98)
# f_cloud = 100
f_hap = 200

# 能耗相关参数
kappa = 0.003  # CPU能效系数
p_s = 2  # 传输功率(W)
p_tx = 2
p_c = 4  
x_l = 0.3  # 本地计算能耗权重  
x_u = 0.3  # UAV传输能耗权重
x_h = 0.4  # HAP传输能耗权重
r_w = 0.5  # 时延和能耗的权衡因子
