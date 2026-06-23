_base_ = ['./efficientnetv2-s_8xb32_in21k.py']

# models setting
model = dict(backbone=dict(arch='m'), )
