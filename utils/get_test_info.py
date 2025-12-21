import re
from collections import defaultdict


def extract_top_dsc_scores(file_path, top_n=30):
    # 读取日志文件
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    # 使用正则表达式匹配文件名和DSC分数
    pattern = r"Image:\s*(.*?\.nii\.gz).*?DSC:\s*(0\.\d{4})"
    matches = re.findall(pattern, log_text, re.DOTALL)

    # 将结果存储在列表中
    results = []
    for filename, dsc in matches:
        results.append({"filename": filename.strip(), "DSC": float(dsc)})

    # 按DSC分数降序排序
    results_sorted = sorted(results, key=lambda x: x["DSC"], reverse=True)

    # 取前top_n个结果
    top_results = results_sorted[:top_n]

    return top_results


# 使用示例
if __name__ == "__main__":
    log_file = (
        # "/data7/zzh/private_train/new_couinaud_VesselEnhancedNet2/test_outputs.txt"
        # "/data7/zzh/public_train/new_couinaud_VesselEnhancedNet2/test_outputs.txt"
        # "/data7/zzh/mixed_train/vessel_HCFormer/test_outputs.txt"
        # "/data7/zzh/private_train/new_couinaud_VesselEnhancedNet_contrastLoss_CATS3/test_outputs.txt"
        # "/data7/zzh/public_train/new_couinaud_VesselEnhancedNet_contrastLoss_CATS3/test_outputs.txt"
        "/data7/zzh/mixed_train/vessel_HCFormer/test_outputs.txt"
    )

    top_30 = extract_top_dsc_scores(log_file, top_n=30)

    # 打印结果
    print(f"{'序号':<5}{'Dice分数':<10}{'文件名'}")
    print("=" * 60)
    for idx, item in enumerate(top_30, 1):
        print(f"{idx:<5}{item['DSC']:<10.4f}{item['filename']}")

    # 可选：保存到CSV文件
    with open("/home/zhaozihao/top_dice_scores.csv", "w", encoding="utf-8") as f:
        f.write("序号,Dice分数,文件名\n")
        for idx, item in enumerate(top_30, 1):
            f.write(f"{idx},{item['DSC']:.4f},{item['filename']}\n")
    print("\n结果已保存到 top_dice_scores.csv")
