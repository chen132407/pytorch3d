## ***********
## Author: tiger
## Date: 2023-07-21 00:57:37
## FilePath: \AI_github_example\CV_git\pytorch3d-main\pytorch3d\docs\tutorials\UseWgetLoadFile.py
## ************

import wget
import os

def loadtest():
    url = 'https://raw.githubusercontent.com/facebookresearch/DensePose/master/DensePoseData/demo_data/texture_from_SURREAL.png'
    url2 = 'https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz'
    url3 = 'https://dl.fbaipublicfiles.com/pytorch3d/data/dolphin/dolphin.obj' # 二进制文件 使用web下载
    # 获取文件名
    res_name = wget.filename_from_url(url3)
    res_name = 'dolphin_obj.txt'
    print(res_name)  #1106F5849B0A2A2A03AAD4B14374596C76B2BDAB_w1000_h626.jpg

    # 下载文件，使用默认文件名,结果返回文件名
    # file_name = wget.download(url)
    # print(file_name) #1106F5849B0A2A2A03AAD4B14374596C76B2BDAB_w1000_h626.jpg

    # 下载文件，重新命名输出文件名

    DATA_DIR = "./data"
    
    DIR = "F:/AI_github_example/CV_git/pytorch3d-main/pytorch3d/docs/tutorials/data/DensePose/"
    DIR2 = "F:/AI_github_example/CV_git/pytorch3d-main/pytorch3d/docs/tutorials/data/"
    target_name = os.path.join(DIR2, res_name)
    file_name = wget.download(url, out= target_name)
    print(file_name) 

    pass

if __name__ == "__main__":
    loadtest()
    pass