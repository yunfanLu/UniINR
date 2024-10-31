import matplotlib.pyplot as plt


scale = 1.1

frame_factors = list(range(0, 9))
frame_factors = [
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
]

our_color = [
    26.16436488,
    26.96881885,
    27.72926018,
    28.78121951,
    29.57610555,
    29.44923345,
    28.56959235,
    27.6985961,
    26.4964383,
]

our_gray = [
    28.5105313,
    29.51070091,
    30.47727905,
    31.45079529,
    32.36187194,
    32.00832588,
    31.30948548,
    30.45530106,
    29.19341865,
]

defeblur_sr_gray = [
    18.80300786,
    17.83183088,
    19.66458446,
    20.16003725,
    20.27522819,
    21.52107088,
    19.83241521,
    18.98698369,
    12.6644299,
]

EvUnroll_TimeLens_Color = [
    26.18,
    22.37474867,
    20.22274979,
    18.99079748,
    19.65535809,
    19.95694998,
    20.53422275,
    21.31973056,
    21.83840345,
]

# 创建折线图
plt.figure(figsize=(8, 5))  # 设置图形大小

# 绘制折线图

plt.plot(
    frame_factors,
    our_gray,
    marker="o",
    linestyle="dotted",
    color="lightseagreen",
    label="UniINR (Gray)",
)


plt.plot(
    frame_factors,
    our_color,
    marker="o",
    linestyle="-",
    color="blue",
    label="UniINR (Color)",
)

plt.plot(
    frame_factors,
    defeblur_sr_gray,
    marker="v",
    linestyle="dotted",
    color="indianred",
    label="DeblurSR (Gray)",
)

plt.plot(
    frame_factors,
    EvUnroll_TimeLens_Color,
    marker="v",
    linestyle="-",
    color="darkorange",
    label="EvUnroll+TimeLens (Color)",
)

plt.xlabel("Different Timestamp t/T", fontsize=scale * 18)  # 设置横轴标签
plt.ylabel("PSNR (dB)", fontsize=scale * 18)  # 设置竖轴标签
plt.title("RS correction, Deblur, and VFI with 9x", fontsize=scale * 20)  # 设置图标题
plt.grid(True)  # 显示网格
plt.legend()  # 显示图例

plt.xticks(frame_factors, fontsize=scale * 12)  # 设置横轴刻度为插帧倍数
plt.yticks(range(11, 35, 2), fontsize=scale * 12)  # 设置竖轴刻度


plt.tight_layout()
# 保存图像
plt.savefig("line_chart.pdf", format="pdf", dpi=600)
plt.savefig("line_chart.png", dpi=600)
# 关闭图形窗口
plt.close()
