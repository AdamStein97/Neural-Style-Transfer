import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import time
from PIL import Image

from utils import load_img, imshow
from StyleContentModel import StyleContentModel


class ImageGenerator():
  def __init__(self, content_image, style_image, style_weight, content_weight, extractor):
    self.extractor = extractor
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.gen_targets(content_image, style_image)
    
  def gen_targets(self, content_image, style_image):
    self.target_content = self.extractor(tf.constant(content_image))['content']
    self.target_style = self.extractor(tf.constant(style_image))['style']
    
  def calc_loss(self, outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.target_style[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= self.style_weight / len(extractor.style_layers)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.target_content[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= self.content_weight / len(extractor.content_layers)
    loss = style_loss + content_loss
    return loss
    
  #@tf.function()
  def train_step(self, image):
    with tf.GradientTape() as tape:
      outputs = self.extractor(image)
      loss = self.calc_loss(outputs)

    grad = tape.gradient(loss, image)
    self.opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return loss
  
  def train(self, epochs, lr, save_dir):
    steps_per_epoch = 100

    generated_image = tf.Variable(content_image)
    self.opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.99, epsilon=1e-1)

    start = time.time()

    step = 0
    
    
    for n in range(epochs):
      for m in range(steps_per_epoch):
        step += 1
        loss = self.train_step(generated_image)
        print(".", end='')
      print('\n')
      tf.keras.backend.print_tensor(loss, message="Loss: ")
      imshow(generated_image.read_value(), "Train step: {}".format(step))
      plt.show()
      
    image_array = generated_image.read_value().numpy()*255
    image_array = np.squeeze(image_array.astype(int),0)

    save_img=Image.fromarray(image_array.astype('uint8'))
    save_img.save(save_dir)
    end = time.time()
    print("Total time: {:.1f}".format(end-start))


# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
		 'block5_conv1'
               ]

tf.enable_eager_execution()

extractor = StyleContentModel(style_layers, content_layers)

content_dir = 'images/turtle.jpg'
style_dir = 'images/wave.jpg'
save_dir = 'images/wave_turtle.jpg'

content_image = load_img(content_dir)
style_image = load_img(style_dir)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

plt.show()

gen = ImageGenerator(content_image, style_image, 1e-2,1e4, extractor)
gen.train(10, 0.02, save_dir)

