# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

import os

# content= 'source_compressed/512/sea_512.jpg'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']



content= 'source_compressed/256/sea.jpg'    # ''#
content_string = '--content ' if content != '' else ''
texture_or_not = 'texture' if content == '' else 'content'
#contents= ['source_compressed/my256/%d.jpg' % i for i in range(1,17)]
contents_name = 'new_loss'# '256'
# styles = ['style_compressed/van_gogh/self_portrait.jpg', 'style_compressed/van_gogh/starry_sky.jpg']
# styles = ['style_compressed/van_gogh/starry_sky256.jpg']
# style_name = 'van_gogh_starry_sky'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']

styles = ['van_gogh/style256.jpg']
style_name = 'van_gogh_starry_sky'

learning_rate=1
iterations=2000
batch_size=1
style_weight=1
content_weight=5
checkpoint_iterations=100
print_iteration = 100
use_mrf = False
use_mrf_string = '--use_mrf' if use_mrf else ''

use_semantic_masks = False
use_semantic_masks_string = '--use_semantic_masks' if use_semantic_masks else ''

output_semantic_mask = 'van_gogh/output_mask256.jpg'
style_semantic_masks = ['van_gogh/style_mask256.jpg']

checkpoint_output='output_checkpoint/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d_%%s.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)
output='output/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/neural_style.py %s%s --styles %s --learning-rate=%f --iterations=%d %s %s --output_semantic_mask=%s --style_semantic_masks %s --style-weight=%d --content-weight=%d --checkpoint-iterations=%d --checkpoint-output=%s --output=%s --print-iterations=%d'
          % (content_string, content, ' '.join(styles), learning_rate, iterations, use_mrf_string, use_semantic_masks_string ,output_semantic_mask, ' '.join(style_semantic_masks), style_weight, content_weight, checkpoint_iterations, checkpoint_output, output, print_iteration))