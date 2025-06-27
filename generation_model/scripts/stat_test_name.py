import torch

# 设置输入 .pt 文件和输出 .txt 文件的路径
pt_file = "/share2/fengbin/A/MOFDiff_CSP/data/CGmof_50/test.pt"
txt_file = "/share2/fengbin/A/MOFDiff_CSP/scripts/test_name.txt"

# 加载 .pt 文件，假设其中的数据是一个字典列表
data = torch.load(pt_file)

# 打开输出文件，逐行写入每个元素的 "mof_id"
with open(txt_file, "w", encoding="utf-8") as f:
    for element in data:
        # 如果字典中包含 "mof_id" 键，则写入对应的值
        if "mof_id" in element:
            f.write(str(element["mof_id"]) + "\n")
        else:
            # 如果没有该键，可以选择跳过或者记录错误
            print("Warning: 当前元素没有 'mof_id' 属性。")
