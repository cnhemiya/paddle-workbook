#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import paddlex as pdx


# 预测图像
image_name = "./dataset/0084.jpg"
# 读取模型
model = pdx.load_model("./output/best_model")
# score阈值，将Box置信度低于该阈值的框过滤
threshold = 0.5

# 读取图像
img = cv2.imread(image_name)
# 预测结果
result = model.predict(img)
# 保留的结果
keep_results = []
# 面积
areas = []
# 写入文件的结果
result_lines = []
# 数量计数
count = 0

# 遍历结果，过滤
for det in result:
    cname, bbox, score = det["category"], det["bbox"], det["score"]
    # 结果过滤
    if score >= threshold:
        count += 1
        keep_results.append(det)
        result_lines.append("{}\n".format(str(det)))
        # 面积：宽 * 高
        areas.append(bbox[2] * bbox[3]) 

# 面积降序排列
# areas = np.asarray(areas)
# sorted_idxs = np.argsort(-areas).tolist()
# keep_results = [keep_results[k]
#                 for k in sorted_idxs] if len(keep_results) > 0 else []

# 符合阈值 threshold 的结果数量
total_str = "the total number is : {}".format(str(count))
print(total_str)

# 写入结果
with open("./result/result.txt", "w") as f:
    f.writelines(result_lines)

# 预测结果可视化
pdx.det.visualize(
    image_name, result, threshold=threshold, save_dir="./result")
