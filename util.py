import yaml


def read_prompt_from_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data.get('prompt', '')
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"错误：解析YAML文件时出错 - {e}")
        return None
