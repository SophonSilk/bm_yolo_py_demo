""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sys
import os
import argparse
import json
import numpy as np
import sophon.sail as sail
import ctypes
import struct
import time
import cv2

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.ab = [x * scale for x in [1, 0, 1, 0, 1, 0]]

  def process(self, input, output, height, width):
    """ Execution function of preprocessing.
    Args:
      input: sail.BMImage, input image
      output: sail.BMImage, output data

    Returns:
      None
    """

    self.bmcv.vpp_resize(input, output,  width, height)

class Net:
  input_shapes_ = {}
  output_shapes_ = {}
  input_tensors_ = {}
  output_tensors_ = {}
  post_process_inputs_ = []
  output_names_ = []
  output_shapes_array_ = []
  preprocessor_ = 0
  tpu_id_ = 0
  handle_ = 0
  img_dtype_ = 0
  engine_ = 0
  graph_name_ = 0
  bmcv_ = 0
  input_name_ = 0
  lib_post_process_ = 0
  input_dtype_ = 0

  def __init__(self, bmodel_path, tpu_id):
    # init Engine
    Net.engine_ = sail.Engine(tpu_id)
    # load bmodel without builtin input and output tensors
    Net.engine_.load(bmodel_path)
    # get model info
    # only one model loaded for this engine
    # only one input tensor and only one output tensor in this graph
    Net.handle_ = Net.engine_.get_handle()
    Net.graph_name_ = Net.engine_.get_graph_names()[0]
    input_names = Net.engine_.get_input_names(Net.graph_name_)
    input_dtype = 0
    Net.tpu_id_ = tpu_id
    Net.input_name_ = input_names[0]
    for i in range(len(input_names)): 
      Net.input_shapes_[input_names[i]] = Net.engine_.get_input_shape(Net.graph_name_, input_names[i])
      input_dtype = Net.engine_.get_input_dtype(Net.graph_name_, input_names[i])
      input = sail.Tensor(Net.handle_, Net.input_shapes_[input_names[i]], input_dtype, False, False)
      Net.input_tensors_[input_names[i]] = input
      Net.input_dtype_ = input_dtype
    Net.output_names_ = Net.engine_.get_output_names(Net.graph_name_)
    for i in range(len(Net.output_names_)): 
      Net.output_shapes_[Net.output_names_[i]] = Net.engine_.get_output_shape(Net.graph_name_, Net.output_names_[i])
      output_dtype = Net.engine_.get_output_dtype(Net.graph_name_, Net.output_names_[i])
      print (Net.output_shapes_[Net.output_names_[i]])
      output = sail.Tensor(Net.handle_, Net.output_shapes_[Net.output_names_[i]], output_dtype, True, True)
      Net.output_tensors_[Net.output_names_[i]] = output
      for j in range(4): 
        Net.output_shapes_array_.append(Net.output_shapes_[Net.output_names_[i]][j])
    print (Net.input_shapes_)
    print (Net.output_shapes_)

    # set io_mode
    Net.engine_.set_io_mode(Net.graph_name_, sail.IOMode.SYSIO)
    Net.bmcv_ = sail.Bmcv(Net.handle_)
    Net.img_dtype_ = Net.bmcv_.get_bm_image_data_format(input_dtype)
    scale = Net.engine_.get_input_scale(Net.graph_name_, input_names[0])
    scale *= 0.003922
    Net.preprocessor_ = PreProcessor(Net.bmcv_, scale)

    # load postprocess so 
    ll = ctypes.cdll.LoadLibrary
    Net.lib_post_process_ = ll('./libYoloPostProcess.so')

    if os.path.exists('result_imgs') is False:
      os.system('mkdir -p result_imgs')

  def cut(obj, sec):
    return [obj[i : i + sec] for i in range(0, len(obj), sec)]

  def dis_image(img):
    dis_img = sail.BMImage(Net.handle_, img.height(),
                        img.width(),
                        sail.Format.FORMAT_BGR_PLANAR, img.dtype())
    Net.bmcv_.vpp_resize(img, dis_img, img.width(), img.height())
    t_img_tensor = sail.Tensor(Net.handle_, [1, 3, img.height(), img.width()], sail.Dtype.BM_UINT8, True, False)
    Net.bmcv_.bm_image_to_tensor(dis_img, t_img_tensor)
    t_img_tensor.sync_d2s()
    np_t_img_tensor = t_img_tensor.asnumpy()
    np_t_img_tensor = np_t_img_tensor.transpose((0, 2, 3, 1))
    np_t_img_tensor = np_t_img_tensor.reshape([img.height(), img.width(), 3])
    cv2.imshow('det_result', np.uint8(np_t_img_tensor))
    cv2.waitKey(10)

  def detect(self, video_path):
    # open a video to be decoded
    decoder = sail.Decoder(video_path, True, Net.tpu_id_)
    frame_id = 0
    while 1:
      img = sail.BMImage()
      # decode a frame from video
      ret = decoder.read(Net.handle_, img)
      if ret != 0:
        print("Finished to read the video!");
        return

      # preprocess image for inference
      tmp = sail.BMImage(Net.handle_, Net.input_shapes_[Net.input_name_][2],
                        Net.input_shapes_[Net.input_name_][3],
                        sail.Format.FORMAT_RGB_PLANAR, img.dtype())
      img_proceesed = sail.BMImage(Net.handle_, Net.input_shapes_[Net.input_name_][2],
                        Net.input_shapes_[Net.input_name_][3],
                        sail.Format.FORMAT_RGB_PLANAR, Net.img_dtype_)
      Net.preprocessor_.process(img,
          tmp, Net.input_shapes_[Net.input_name_][2], Net.input_shapes_[Net.input_name_][3])

      Net.bmcv_.convert_to(tmp, img_proceesed, ((Net.preprocessor_.ab[0], Net.preprocessor_.ab[1]), \
                                                           (Net.preprocessor_.ab[2], Net.preprocessor_.ab[3]), \
                                                           (Net.preprocessor_.ab[4], Net.preprocessor_.ab[5])))
      Net.bmcv_.bm_image_to_tensor(img_proceesed, Net.input_tensors_[Net.input_name_])

      # do inference 
      Net.engine_.process(Net.graph_name_,
              Net.input_tensors_, Net.input_shapes_, Net.output_tensors_)

      # post process
      # set param for diffrent model
      class_num = 80
      score_threshold = 0.5
      top_k = 200
      anchor_biases = np.array([12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401], dtype = np.float32)
      anchor_biases_len = anchor_biases.shape[0]
      anchor_biases = anchor_biases.ctypes.data_as(ctypes.c_char_p)

      masks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = np.int32)
      masks = masks.ctypes.data_as(ctypes.c_char_p)
      # end set param for diffrent model

      net_shape = np.array(Net.input_shapes_[Net.input_name_])
      net_shape = net_shape.astype('int32')
      net_shape = net_shape.ctypes.data_as(ctypes.c_char_p)

      output_tensor_num = len(Net.output_tensors_)
      output_shape = np.array(Net.output_shapes_array_)
      output_shape = output_shape.astype('int32')
      output_shape = output_shape.ctypes.data_as(ctypes.c_char_p)

      CLONG_P_INPUT = len(Net.output_tensors_) * ctypes.c_long
      Net.post_process_inputs_ = CLONG_P_INPUT()
      for i in range(len(Net.output_tensors_)):
        output_data = Net.output_tensors_[Net.output_names_[i]].pysys_data()
        Net.post_process_inputs_[i] = output_data[0]

      dets = ctypes.create_string_buffer(4 * 7 * top_k)
      obj_num = ctypes.create_string_buffer(4)

      Net.lib_post_process_.process(Net.post_process_inputs_,
                     net_shape, output_shape, output_tensor_num,
                     ctypes.c_float(score_threshold),
                     class_num, anchor_biases, masks, anchor_biases_len,
                     img.width(), img.height(), top_k, dets, obj_num)

      # get the detect results
      # batch_id, class_id, score, left, top, right, bottom
      i_obj_num = Net.cut(obj_num, 4)
      i_obj_num = struct.unpack('<i', struct.pack('4B', *i_obj_num[0]))[0]
      i_obj_num = int(i_obj_num)
      dets = Net.cut(dets, 4)

      # draw detect results
      for i in range(0, i_obj_num * 7, 7): 
        class_id = struct.unpack('<f', struct.pack('4B', *dets[i + 1]))[0]
        score = struct.unpack('<f', struct.pack('4B', *dets[i + 2]))[0]
        left = struct.unpack('<f', struct.pack('4B', *dets[i + 3]))[0]
        top = struct.unpack('<f', struct.pack('4B', *dets[i + 4]))[0]
        right = struct.unpack('<f', struct.pack('4B', *dets[i + 5]))[0]
        bottom = struct.unpack('<f', struct.pack('4B', *dets[i + 6]))[0]
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        Net.bmcv_.rectangle(img, left, top, right - left + 1, bottom - top + 1, (0, 255, 0), 3)
      save_full_path = os.path.join('result_imgs', str(frame_id) + '_video.jpg')
      Net.bmcv_.imwrite(save_full_path, img)
      frame_id += 1

if __name__ == '__main__':
  """ A Yolo example using bm-ffmpeg to decode and bmcv to preprocess.
  """
  desc='decode (ffmpeg) + preprocess (bmcv) + inference (sophon inference)'
  PARSER = argparse.ArgumentParser(description=desc)
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  yolo_net = Net(ARGS.bmodel, ARGS.tpu_id)
  yolo_net.detect(ARGS.input)
