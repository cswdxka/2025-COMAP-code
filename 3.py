import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取CSV文件
df= pd.read_csv("C:/Users/dxw/Desktop/1.csv")

# 获取唯一年份
years = df['Year'].unique()

# 存储每年聚类结果的字典
yearly_clusters = {}
# plt.figure(figsize=(12, 8))

# 对每个年份进行k-means聚类
for year in years:
    # 选取该年份的数据
    year_data = df[df['Year'] == year]

    # 进行k-means聚类 (k=3)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(year_data[['Total']])

    # 获取聚类中心
    centers = kmeans.cluster_centers_

    # 获取分界线位置
    sorted_centers = np.sort(centers.flatten())
    boundaries = [(sorted_centers[i] + sorted_centers[i + 1]) / 2 for i in range(len(sorted_centers) - 1)]

    # 存储结果
    yearly_clusters[year] = {'centers': sorted_centers, 'boundaries': boundaries}
    # plt.subplots(figsize = (6,4))

    # 绘制该年的聚类结果
    plt.scatter(year_data['Total'], np.zeros_like(year_data['Total']) + year, label=f'Year {year}')
    plt.plot(sorted_centers, np.ones_like(sorted_centers) * year, 'x', markersize=14, label=f'Centers Year {year}')

    # 可视化分界线
    # for boundary in boundaries:
    #     plt.axvline(x=boundary, linestyle='--', color='gray')
#
# 设置图表样式
plt.title('Scatter plot of total medals using K-means++ clustering')
plt.xlabel('Total')
plt.ylabel('Year')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 调整图像大小
plt.show()

# 输出聚类中心和分界线
# for year, data in yearly_clusters.items():
#     print(f"Year {year}:")
#     print(f"  Clustering Centers: {data['centers']}")
#     print(f"  Boundary Positions: {data['boundaries']}")

for year, data in yearly_clusters.items():
    cluster_centers_str = [f"{center:.2f}" for center in data['centers']]
    boundaries_str = [f"{boundary:.2f}" for boundary in data['boundaries']]

    print(f"{year} [聚类中心: {', '.join(cluster_centers_str)}] [分界线: {', '.join(boundaries_str)}]")