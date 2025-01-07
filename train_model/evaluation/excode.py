import yaml
import numpy as np

# 自定义构造器来处理 NumPy 对象
def numpy_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    dtype = mapping.get('dtype')
    scalar = mapping.get('state')
    if dtype and scalar:
        return np.frombuffer(scalar, dtype=dtype).item()
    return mapping

# 添加自定义构造器到 YAML 加载器
def construct_custom_yaml(loader, node):
    loader.add_constructor('!!python/object/apply:numpy.core.multiarray.scalar', numpy_constructor)

# 文件路径
file_path = 'evaluation_results.yaml'

# 读取YAML文件
with open(file_path, 'r') as file:
    construct_custom_yaml(yaml.Loader, yaml.SafeLoader)  # 添加自定义构造器
    evaluation_results = yaml.safe_load(file)

# 打印解析后的内容
print(evaluation_results)

# 访问特定的值
accuracy = evaluation_results['overall_metrics']['accuracy']
print(f"Overall Accuracy: {accuracy}")

# 如果你需要将numpy对象转换回其原始值，你可以使用以下代码：
accuracy_value = np.frombuffer(evaluation_results['overall_metrics']['accuracy'][1], dtype=evaluation_results['overall_metrics']['accuracy'][0]).item()
print(f"Overall Accuracy Value: {accuracy_value}")