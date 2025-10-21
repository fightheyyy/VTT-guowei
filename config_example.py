"""
TimesCLIP模型配置示例
包含不同场景下的推荐配置
"""


# 配置1: 短期预测（小数据集）
CONFIG_SHORT_TERM = {
    'time_steps': 96,
    'n_variates': 7,
    'prediction_steps': 96,
    'patch_length': 16,
    'stride': 8,
    'd_model': 256,
    'n_heads': 4,
    'temperature': 0.07,
    'image_size': 224,
    'clip_model_name': 'openai/clip-vit-base-patch16',
    
    # 训练参数
    'batch_size': 32,
    'lr_vision': 1e-5,
    'lr_other': 1e-4,
    'lambda_gen': 1.0,
    'lambda_align': 0.1,
    'epochs': 50
}


# 配置2: 长期预测（中等数据集）
CONFIG_LONG_TERM = {
    'time_steps': 336,
    'n_variates': 21,
    'prediction_steps': 720,
    'patch_length': 24,
    'stride': 12,
    'd_model': 512,
    'n_heads': 8,
    'temperature': 0.07,
    'image_size': 224,
    'clip_model_name': 'openai/clip-vit-base-patch16',
    
    # 训练参数
    'batch_size': 16,
    'lr_vision': 5e-6,
    'lr_other': 5e-5,
    'lambda_gen': 1.0,
    'lambda_align': 0.05,
    'epochs': 100
}


# 配置3: 高分辨率（大数据集）
CONFIG_HIGH_RES = {
    'time_steps': 512,
    'n_variates': 50,
    'prediction_steps': 192,
    'patch_length': 32,
    'stride': 16,
    'd_model': 768,
    'n_heads': 12,
    'temperature': 0.05,
    'image_size': 224,
    'clip_model_name': 'openai/clip-vit-base-patch16',
    
    # 训练参数
    'batch_size': 8,
    'lr_vision': 1e-6,
    'lr_other': 1e-5,
    'lambda_gen': 1.0,
    'lambda_align': 0.02,
    'epochs': 150
}


# 配置4: 快速实验（调试用）
CONFIG_DEBUG = {
    'time_steps': 48,
    'n_variates': 3,
    'prediction_steps': 24,
    'patch_length': 8,
    'stride': 4,
    'd_model': 128,
    'n_heads': 4,
    'temperature': 0.1,
    'image_size': 224,
    'clip_model_name': 'openai/clip-vit-base-patch16',
    
    # 训练参数
    'batch_size': 4,
    'lr_pretrained': 1e-5,
    'lr_new': 1e-4,
    'lambda_gen': 1.0,
    'lambda_align': 0.1,
    'epochs': 5
}


def get_config(config_name='short_term'):
    """
    获取指定配置
    
    参数:
        config_name: 配置名称
            - 'short_term': 短期预测
            - 'long_term': 长期预测
            - 'high_res': 高分辨率
            - 'debug': 调试配置
    
    返回:
        配置字典
    """
    configs = {
        'short_term': CONFIG_SHORT_TERM,
        'long_term': CONFIG_LONG_TERM,
        'high_res': CONFIG_HIGH_RES,
        'debug': CONFIG_DEBUG
    }
    
    if config_name not in configs:
        raise ValueError(f"未知配置: {config_name}. 可选: {list(configs.keys())}")
    
    return configs[config_name]


if __name__ == "__main__":
    # 示例：加载配置
    config = get_config('short_term')
    print("短期预测配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

