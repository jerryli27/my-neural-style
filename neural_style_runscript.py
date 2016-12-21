# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

import os

# content= 'source_compressed/512/sea_512.jpg'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']

for style_w in range(7):
    for image_i in range(1,4):

        content= 'source_compressed/IMG_0677_1536.jpg'   # ''#
        content_string = '--content ' if content != '' else ''
        texture_or_not = 'texture' if content == '' else 'content'
        #contents= ['source_compressed/my256/%d.jpg' % i for i in range(1,17)]
        contents_name = 'toru'# '256'
        # styles = ['style_compressed/van_gogh/self_portrait.jpg', 'style_compressed/van_gogh/starry_sky.jpg']
        # styles = ['style_compressed/van_gogh/starry_sky256.jpg']
        # style_name = 'van_gogh_starry_sky'
        # styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']

        styles = ['style_compressed/test_toru/%d_352x1536.jpg' %image_i]
        style_name = 'sunset_%d' %image_i
        use_semantic_masks = False
        use_semantic_masks_string = '--use_semantic_masks' if use_semantic_masks else ''

        use_mrf = True
        use_mrf_string = '--use_mrf' if use_mrf else ''

        learning_rate=10 # larger lr seems to result in larger looking features (compared to lr = 1)
        iterations=1000
        batch_size=1
        style_weight=100 if not use_mrf else (3 + style_w / 2.0) # 100 for old gram loss works. The weight for mrf varies from picture to picture.
        content_weight=5
        checkpoint_iterations=100
        print_iteration = 100
        width = 1536
        height = 352


        output_semantic_mask = 'van_gogh/bw_masks/'
        style_semantic_masks = ['van_gogh/bw_masks/']

        checkpoint_output='output_checkpoint/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d_%%s.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)
        output='output/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)

        # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
        os.system('python ~/PycharmProjects/my-neural-style/neural_style.py %s%s --styles %s --learning-rate=%f --iterations=%d %s %s --output_semantic_mask=%s --style_semantic_masks %s --style-weight=%f --content-weight=%f --checkpoint-iterations=%d --checkpoint-output=%s --output=%s --print-iterations=%d --width=%d --height=%d'
                  % (content_string, content, ' '.join(styles), learning_rate, iterations, use_mrf_string, use_semantic_masks_string ,output_semantic_mask, ' '.join(style_semantic_masks), style_weight, content_weight, checkpoint_iterations, checkpoint_output, output, print_iteration, width, height))



"""
# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

import os

# content= 'source_compressed/512/sea_512.jpg'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']



content= 'source_compressed/256/sea.jpg'   # ''#
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
use_semantic_masks = True
use_semantic_masks_string = '--use_semantic_masks' if use_semantic_masks else ''

use_mrf = False
use_mrf_string = '--use_mrf' if use_mrf else ''

learning_rate=10 # larger lr seems to result in larger looking features (compared to lr = 1)
iterations=1000
batch_size=1
style_weight=100 if not use_mrf else 5 # 100 for old gram loss works.
content_weight=5
checkpoint_iterations=100
print_iteration = 100


output_semantic_mask = 'van_gogh/bw_masks/'
style_semantic_masks = ['van_gogh/bw_masks/']

checkpoint_output='output_checkpoint/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d_%%s.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)
output='output/%s-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-use_masks-%s-style-%d-content-%d.jpg' % (contents_name, style_name, texture_or_not, iterations, batch_size, learning_rate, str(use_mrf), str(use_semantic_masks), style_weight, content_weight)

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/neural_style.py %s%s --styles %s --learning-rate=%f --iterations=%d %s %s --output_semantic_mask=%s --style_semantic_masks %s --style-weight=%d --content-weight=%d --checkpoint-iterations=%d --checkpoint-output=%s --output=%s --print-iterations=%d'
          % (content_string, content, ' '.join(styles), learning_rate, iterations, use_mrf_string, use_semantic_masks_string ,output_semantic_mask, ' '.join(style_semantic_masks), style_weight, content_weight, checkpoint_iterations, checkpoint_output, output, print_iteration))
"""