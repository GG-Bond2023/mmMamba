_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

# models settings
model = dict(backbone=dict(arch='small'), head=dict(in_channels=512))
