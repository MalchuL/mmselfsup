import torch

CKPT_PATH = '/home/malchul/work/self_supervised/mmselfsup/work_dirs/selfsup/005_adam_pretrain_vs_generator_gan_init/epoch_100.pth'
OUT_PATH = '/home/malchul/work/self_supervised/mmselfsup/work_dirs/selfsup/005_adam_pretrain_vs_generator_gan_init/epoch_100_clear.pth'
x = torch.load(CKPT_PATH, map_location='cpu')

del x['meta']
del x['message_hub']
del x['optimizer']
del x['param_schedulers']

state_dict = x['state_dict']
new_state_dict = {k.replace('momentum_encoder.module.0.', 'netG.'): v for k, v in state_dict.items() if
                  'momentum_encoder.module.0' in k}
x['state_dict'] = new_state_dict
torch.save(x, OUT_PATH)