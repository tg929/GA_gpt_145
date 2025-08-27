import matplotlib
import os

# 获取 Matplotlib 缓存目录的路径
cache_dir = matplotlib.get_cachedir()
print(f"Matplotlib 缓存目录位于: {cache_dir}")

# 构造字体缓存文件的路径 (文件名通常是 fontlist-vXXX.json)
# 为了安全起见，我们直接删除目录下的所有 .json 文件，或者整个目录的内容
# 更简单粗暴但有效的方法是直接删除整个缓存目录
# 注意：这只会删除缓存，不会损坏你的 Matplotlib 安装
try:
    # 遍历缓存目录并删除文件
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("成功清除 Matplotlib 字体缓存。")
except Exception as e:
    print(f"清除缓存时出错: {e}")
    print("您可以尝试手动删除上面显示的缓存目录中的所有文件。")