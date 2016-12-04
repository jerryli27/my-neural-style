"""
This file uses the texture nets technique to generate an image by combining style of an input and the content of
another input.
Citations:
A Learned Representation For Artistic Style https://arxiv.org/abs/1610.07629
Texture Networks: Feed-forward Synthesis of Textures and Stylized Images https://arxiv.org/abs/1603.03417
Instance Normalization: The Missing Ingredient for Fast Stylization https://arxiv.org/abs/1607.08022
Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis: https://arxiv.org/abs/1601.04589
Perceptual Losses for Real-Time Style Transfer and Super-Resolution: https://arxiv.org/abs/1603.08155
Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks: https://arxiv.org/abs/1603.01768
"""

from argparse import ArgumentParser

import n_style_feedforward_net
from general_util import *

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2
# lr = 0.001 in https://arxiv.org/abs/1610.07629.
# Higher learning rate than 0.01 may sacrifice the quality of the network.
LEARNING_RATE = 0.001
STYLE_SCALE = 1.0
ITERATIONS = 160000  # 40000 in https://arxiv.org/abs/1610.07629
BATCH_SIZE = 4  # 16 in https://arxiv.org/abs/1610.07629
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
PRINT_ITERATIONS = 100


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_folder', dest='content_folder',
                        help='The path to the content images for training. In the papers they use the Microsoft COCO dataset.',
                        metavar='CONTENT_FOLDER', default='../johnson-fast-neural-style/fast-style-transfer/data/train2014/')

    parser.add_argument('--style_folder', dest='style_folder',
                        help='The path to the style images for training.',
                        metavar='STYLE_FOLDER',
                        default='/home/jerryli27/shirobako01pic/')

    parser.add_argument('--styles',dest='styles', nargs='+',
                        help='One or more style images.',
                        metavar='STYLE', required=True)
    parser.add_argument('--texture_synthesis_only', dest='texture_synthesis_only',
                        help='If true, we only generate the texture of the style images.'
                             ' No content image will be used.',
                        action='store_true')
    parser.set_defaults(texture_synthesis_only=False)
    parser.add_argument('--output', dest='output',
                        help='Output path.',
                        metavar='OUTPUT', required=True)
    parser.add_argument('--checkpoint_output', dest='checkpoint_output',
                        help='The checkpoint output format. It must contain 2 %s, the first one for content index '
                             'and the second one for style index.',
                        metavar='OUTPUT')
    parser.add_argument('--iterations', type=int, dest='iterations',
                        help='Iterations (default %(default)s).',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        help='Batch size (default %(default)s).',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--height', type=int, dest='height',
                        help='Input and output height. All content images and style images should be automatically '
                             'scaled accordingly.',
                        metavar='HEIGHT', default=256)
    parser.add_argument('--width', type=int, dest='width',
                        help='Input and output width. All content images and style images should be automatically '
                             'scaled accordingly.',
                        metavar='WIDTH', default=256)
    parser.add_argument('--network', dest='network',
                        help='Path to network parameters (default %(default)s).',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--use_mrf', dest='use_mrf',
                        help='If true, we use Markov Random Fields loss instead of Gramian loss.'
                             ' (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_mrf=False)
    parser.add_argument('--use_johnson', dest='use_johnson',
                        help='If true, we use the johnson generator net instead of pyramid net (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_johnson=False)
    parser.add_argument('--multiple_styles_train_scale_offset_only', dest='multiple_styles_train_scale_offset_only',
                        help='If true, TODO (default %(default)s).',
                        action='store_true')
    parser.set_defaults(multiple_styles_train_scale_offset_only=False)
    # TODO: delete content weight after we make sure we do not need tv weight.
    parser.add_argument('--content_weight', type=float, dest='content_weight',
                        help='Content weight (default %(default)s).',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style_weight', type=float, dest='style_weight',
                        help='Style weight (default %(default)s).',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style_blend_weights', type=float, dest='style_blend_weights',
                        help='How much we weigh each style during the training. The more we weigh one style, the more '
                             'loss will come from that style and the more the output image will look like that style. '
                             'During training it should not be set because the network automatically deals with '
                             'multiple styles.',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv_weight', type=float, dest='tv_weight',
                        help='Total variation regularization weight (default %(default)s).',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                        help='Learning rate (default %(default)s).',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--print_iterations', type=int, dest='print_iterations',
                        help='Statistics printing frequency.',
                        metavar='PRINT_ITERATIONS', default=PRINT_ITERATIONS)
    parser.add_argument('--checkpoint_iterations', type=int, dest='checkpoint_iterations',
                        help='Checkpoint frequency.',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--test_img', type=str,
                        dest='test_img', help='test image path',
                        metavar='TEST_IMAGE')
    parser.add_argument('--model_save_dir', dest='model_save_dir',
                        help='The directory to save trained model and its checkpoints.',
                        metavar='MODEL_SAVE_DIR', default='models/')
    parser.add_argument('--do_restore_and_generate', type=bool, dest='do_restore_and_generate',
                        help='If true, it generates an image from a previously trained model. '
                             'Otherwise it does training and generate a model.',
                        metavar='DO_RESTORE_AND_GENERATE', default=False)
    parser.add_argument('--do_restore_and_train', dest='do_restore_and_train',
                        help='If set, we read the model at model_save_dir and start training from there. '
                             'The overall setting and structure must be the same.',
                        action='store_true')
    parser.set_defaults(do_restore_and_train=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    style_images = read_and_resize_images(options.styles, options.height, options.width)

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # Default is equal weights. There is no need to divide weight by number of styles, because at training time,
        # for each style, we do one content loss training and one style loss training. If we do the division, then
        # it favors the content loss by a factor of number of styles.
        style_blend_weights = [1.0 for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight * len(style_blend_weights)
                               for weight in style_blend_weights]

    if options.output and options.output.count("%s") != 1:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain only one `%s` (e.g. `foo_style_%s.jpg`).")
    if options.checkpoint_output and options.checkpoint_output.count("%s") != 2:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain only two `%s` (e.g. `foo_style_%s_iteration_%s.jpg`).")

    for iteration, image in n_style_feedforward_net.style_synthesis_net(
            path_to_network=options.network,
            height=options.height,
            width=options.width,
            styles=style_images,
            iterations=options.iterations,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            style_weight=options.style_weight,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            style_only=options.texture_synthesis_only,
            multiple_styles_train_scale_offset_only= options.multiple_styles_train_scale_offset_only,
            use_mrf=options.use_mrf,
            use_johnson=options.use_johnson,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
            save_dir=options.model_save_dir,
            do_restore_and_generate=options.do_restore_and_generate,
            do_restore_and_train=options.do_restore_and_train,
            content_folder=options.content_folder,
            test_img_dir=options.test_img
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image)
        else:
            for style_i, _ in enumerate(options.styles):
                if options.test_img:
                    if iteration is not None:
                        output_file = options.checkpoint_output % (style_i, iteration)
                    else:
                        output_file = options.output % (style_i)  # TODO: add test for legal output.
                    if output_file:
                        imsave(output_file, image[style_i])

if __name__ == '__main__':
    main()
    # following are some lists of possible commands.
    """
    --content=source_compressed/512/sea_512.jpg --styles style_compressed/claude_monet/512/1.jpg --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s.jpg --learning_rate=10 --iterations=1500 --style_weight=100 --content_weight=5 --checkpoint_output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s_%s.jpg" --checkpoint_iterations=300
    """

# TODO:
"""
It is crucial to have a low enough learning rate. If you have a high one and decrease it over time, it still won't work.

MRF is confirmed not to work with feed forward neural network. But mrf is not really absolutely necessary. I might need
it for semantic related masking stuff. But it has also been shown that masking works with gram loss and feed forward network.

Here are two unsolved problems:
The size of the features/styles. The image genearted when input is 100 x 100 versus 1000 x 1000 is drastically different.
The generated image is limited to the structure of the original painting. In art, you can draw something larger or
smaller than usual to achieve some effect. There's no such thing here because the loss is localized
(You can't just change the absolute position of objects in content image.)


Two ideas: artists "zoom in" to a spot when they want to paint it in more detail. Is it possible to do so in our architecture?

Second idea: if the generator network can generate styled images, why can't it be used to reformat the feature layers?
Turn the feature layers into something that look like other artforms (without loosing too much originality)
and generate ... But how's that different from just use the generator network to generate the image? Not really

I want something other than difference squared loss. I thought about nearest neighbor loss of the normalized feature
layers. That one is invariant to translation.


Trying two things now: one is to directly observe what is going on in each loss function for each feature layer.
Another is to feed multiple styles into the network while modifying the same parameters.

simply feed 15000 shirobako images as style didn't work.

I tried the mask again using hand-segmented shirobako mask (mrf loss). Surprisingly it's sort of working. Maybe I don't need the
content loss. I can just create a drawing from scratch. Maybe that would be too hard.
One more difficulty if I choose to go on this direction. I don't have hand-segmented masks for me to train.
Even if I do, I don't know how to incorporate say 10 masked style images together. Simple nearest neighbor matching on
the 10 images would require 10x memory in the gpu.

One idea for automatically label images: start with example image and a hand-labeled mask. Now for each input
we compute the nearest neighbor of say conv4-2 (any high level layer) and assign the nearest neighbor's mask
to the input layer. Then add the constraint that nearby pixels should preferrably have the same label.
Then do deconv to get pixel-wise label for the original image. (Sounds like a plan, but this requires the
high level layers of the style and content to be similar, which may well not be the case. 20161122)

Texture and placement of objects are two different things. Texture is given a mask, how to fill in the colors
so that the style loss is the least. That is already solved. Placement of object is another issue. How to
place objects relative to each other so that it is the most probable.

I tried the nn loss. NN is not differentiable so it has no gradient. I should've realized earlier. Now I probably have
to get around this problem.

One thing is, the reason why mrf can perform better is because it has those patches that captures info about nearby
cells. Gram simply multiply them together and add all cells. There's no interaction between a cell and the cell on its
side. On the otherhand, the mrf is expensive to calculate because of the nn step.

After some thinking, it comes down to knowing the feature layers first. (like each of the conv layers, when they're
activated and compare two inputs, content and style, see when their conv layers look similar.)

Adversarial generative network may be worth investigating into. But don't do that just yet. I still have to finish up
this experiment.

We can make the additional semantic masks learnable. Just add a loss for too much deviation from original (more
complicated loss can be added later)

Things learned from the experiment: the conv layer won't look close to the original image. So we can't hope to morph
one image into another in the conv layers. Also, making semantic masks learnable will need further refinement.

I thought maybe finding the spatial correlation of semantic layers might be helpful. Now it is only finding the
correlation in-place (that is, the same pixel times the same pixel in another layer). What happens if we shift it by
say half of the width and find the correlation? The hope is that , for example we have two eyes, one on the left
and one on the right. By finding the correlation after shifting, we may find: ah whenever there's an
eye here, there will be on on its right with distance half screen away. That's my hope.


I was correct that correlation after shifting one layer encodes the relative positional information. Now the problem is
1. It was not perfect. I hope the problem can be solved after adding a content loss.
2. It was spatially too similar to the style image. ie. no shifting etc. There was some shift for subfeatures, like
the mouth was on the right first and slowly came to the center. But overall the head position was still the same, the
position of everything was the same as in the style. This is not what we want. We just want to modify the content image
a little so that the eyes become bigger, or the nose become less noticeable and things like that.
I want to have two very simple images as input and see how it goes.

Not going well so far.

Side projects during thxgiving:
Run multi-style feed forward network.
Shit.. I realized it actually doesn't make sense in the current frame work. I have to separate each style by itself
instead of feeding them in as batches. ( Otherwise I can't set the image placeholder so I can't train scale and offset
individually. I'll try to fix this tomorrow by going back to the master branch and merge the two..
I don't know if I should keep the batch style... Maybe I should.

Tested overnight on claude monet's paintings. Now I'm sure I can't just merge styles like that. It doesn't work.

Other future directions: feed forward neural doodle.


As I think of it, maybe we don't need the content image at all? There are many things that are hard to find in real life
but exist in drawings. I know that a feed forward network with masks can be trained as long as I have enough training
data. The only problem is where to get that semantically labeled training data. I can get that from 3d model... I think
I can look into that direction. No content image. Content loss comes directly from thousands of style images. Details
of how content loss can come from those may be filled in later but I think this is the right direction. We don't have
to worry about distorting content image into the style image, which is a big challenge.

Before that, let me modify the current code so that it can accept only semantic style images and semantic masks and
learn them using feed-forward nn. the loss... we can use content loss or style loss. try both I guess. I need to
copy the previous code into a new file because the change will make the code incompatible with the previous version.
(since there no longer need to be a content image). Actually, I can use the same file... Since I may use that framework
to learn auto generating textures.

Basically I will implement feed forward neural doodle, but better than what they've done. I need not only rgb mask, but
any number of masks for each style image.
"""