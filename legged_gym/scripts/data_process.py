import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- 核心物理参数 (需对应你的 Isaac Gym 台阶设置) ---
STAIR_TREAD = 0.3    # 台阶踏面宽度 (m)
DANGER_ZONE = 0.05   # 距离边缘 5cm 视为危险区域

def find_project_root():
    """ 自动寻找包含 logs 文件夹的项目根目录 """
    curr_path = os.path.abspath(os.getcwd())
    # 向上追溯，直到找到包含 logs 目录的文件夹
    while curr_path != os.path.dirname(curr_path):
        if os.path.exists(os.path.join(curr_path, 'logs')):
            return curr_path
        curr_path = os.path.dirname(curr_path)
    return None

def load_all_data(root_path):
    """ 
    根据 tree.txt 显示的结构递归加载所有 CSV 
    支持 logs/g1/data/ 下的所有文件
    """
    # 构造搜索模式，匹配所有符合命名的文件
    data_dir = "/home/ris/111/logs/g1/data/Jan25_22-26-05"
    search_pattern = os.path.join(data_dir, "footholds_iter_*.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"❌ 错误: 未在 {os.path.join(root_path, 'logs/g1/data/')} 找到 CSV 数据！")
        return None
    
    print(f"✅ 成功找到 {len(files)} 个数据文件，正在聚合千万级数据点...")
    
    # 使用列表推导式高效加载
    df_list = []
    for i, f in enumerate(files):
        try:
            temp_df = pd.read_csv(f)
            if not temp_df.empty:
                df_list.append(temp_df)
            if i % 20 == 0:
                print(f"   已读取 {i}/{len(files)} 个文件...")
        except Exception:
            continue
            
    return pd.concat(df_list, ignore_index=True) if df_list else None

def run_professional_analysis():
    root = find_project_root()
    if not root:
        print("❌ 错误: 找不到项目根目录！请确保在 unitree_rl_gym-main 文件夹内运行。")
        return

    df = load_all_data(root)
    if df is None: return

    # 1. 坐标投影：将世界坐标映射到 0.0 ~ 0.3 的台阶踏面上
    df['rel_x'] = df['x'] % STAIR_TREAD
    df['rel_y'] = df['y']

    # 2. 定量指标计算 (填入论文 Table 1)
    mean_dev = (df['rel_x'] - (STAIR_TREAD/2)).abs().mean() * 100 
    fail_rate = (df['rel_x'] < DANGER_ZONE).mean() * 100

    print(f"\n" + "="*45)
    print(f" 📊 盲视基准组 (Blind Locomotion) 统计结果 ")
    print(f"="*45)
    print(f"总计记录点数: {len(df):,}")
    print(f"平均落足偏差: {mean_dev:.2f} cm")
    print(f"边缘失败率:   {fail_rate:.2f}% (危险区 < {DANGER_ZONE*100}cm)")
    print(f"="*45)

    # 3. 绘图 (双子图布局：密度分布图 + 热力图)
    print("🎨 正在生成学术级图表，请稍候...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3)

    # --- 左图：密度分布热力图 (Hexbin) ---
    # gridsize 决定网格精细度，cmap 使用黄红渐变突出高频区
    hb = ax1.hexbin(df['rel_x'], df['rel_y'], gridsize=50, cmap='YlOrRd', mincnt=1)
    cb = fig.colorbar(hb, ax=ax1)
    cb.set_label('Contact Frequency (Counts)')
    
    ax1.axvspan(0, DANGER_ZONE, color='red', alpha=0.1, label='Danger Zone (Edge)')
    ax1.axvline(x=DANGER_ZONE, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title("A. Spatial Contact Density (Heatmap)", fontsize=13)
    ax1.set_xlabel("Tread Depth (m) [0.0 = Edge]", fontsize=11)
    ax1.set_ylabel("Lateral Position (m)", fontsize=11)
    ax1.set_xlim(0, STAIR_TREAD)
    ax1.legend(loc='upper right')

    # --- 右图：边缘距离概率分布 (Histogram + Trend) ---
    ax2.hist(df['rel_x'], bins=60, density=True, color='skyblue', alpha=0.7, edgecolor='white')
    # 抽样一部分数据绘制趋势线（千万级全画会很慢，10万点采样足够反映趋势）
    sample_data = df['rel_x'].sample(min(100000, len(df)))
    sample_data.plot.kde(ax=ax2, color='darkblue', linewidth=2, label='Distribution Trend')
    
    ax2.axvline(x=DANGER_ZONE, color='red', linestyle='--', label='Edge Limit')
    ax2.set_title("B. Foothold Probability Density", fontsize=13)
    ax2.set_xlabel("Distance from Edge (m)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_xlim(0, STAIR_TREAD)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.legend()

    # 4. 保存文件 (同时保存 PNG 用于查看和 PDF 用于论文)
    # png_path = os.path.join(root, "blind_analysis_report.png")
    # pdf_path = os.path.join(root, "blind_analysis_report.pdf")
    png_path = os.path.join(root, "pure_vision_analysis_report.png")
    pdf_path = os.path.join(root, "pure_vision_analysis_report.pdf")
    # png_path = os.path.join(root, "vision_analysis_report.png")
    # pdf_path = os.path.join(root, "vision_analysis_report.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\n✨ 任务完成！")
    print(f"🖼️  预览图已保存: {png_path}")
    print(f"📄 论文专用PDF已保存: {pdf_path}")
    
    # 在 Headless 环境下运行可能会报错，如果在服务器上跑请注释掉 plt.show()
    # plt.show()

if __name__ == "__main__":
    run_professional_analysis()