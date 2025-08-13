import os
import requests

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} 已存在，跳过下载。")
        return
    print(f"正在下载 {url} 到 {save_path} ...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"下载完成：{save_path}")

def main():
    # 数据集下载链接
    dataset_urls = {
        # "CAVE": "http://www.cs.columbia.edu/CAVE/databases/multispectral/zip/CAVE_1_1.zip",
        "PaviaU": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University",
        "IndianPines": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        # "Harvard": "https://vision.seas.harvard.edu/hyperspec/hyperspectral_FoV1.zip",
        # "Chikusei": "https://www.eorc.jaxa.jp/ALOS/en/dataset/chikusei_dataset.zip"
    }
    save_names = {
        # "CAVE": "CAVE.zip",
        "PaviaU": "PaviaU.mat",
        "IndianPines": "IndianPines.mat",
        # "Harvard": "Harvard.zip",
        # "Chikusei": "Chikusei.zip"
    }
    for name, url in dataset_urls.items():
        save_path = os.path.join(os.path.dirname(__file__), save_names[name])
        download_file(url, save_path)
    print("部分数据集（如CAVE、Harvard、Chikusei）需要手动在官网下载或申请，无法自动脚本下载。")
    print("请访问各数据集官网，下载后放到本目录下对应文件夹。")

if __name__ == "__main__":
    main()