import argparse
import face_model
import cv2
import sys
import os
import functools
import operator
import h5py
import numpy as np
from multiprocessing import Pool

def file_number(number, string_length):
    """Generate number strings with leading zeros so that numerical and string order coincide"""
    number_to_string = str(number+1)
    length = len(number_to_string)
    final_name = '0'*(string_length-length) + number_to_string
    return final_name

def range_values_and_batches(arguments, total_number_arg):
    batch_size = int(arguments.batch_size)
    end_values = list(range(0, total_number_arg, batch_size)[1:])
    end_values.append(total_number_arg)
    start_values = list(range(0, total_number_arg, batch_size))
    length = len(start_values)
    return [(arguments, start_values.pop(0), end_values.pop(0), i) for i in range(length)]

class EmbeddingGenerator(object):
  """This class constructs face vectors for each image in the dataset."""
  def __init__(self, args):
    self.image_directory_path = args.image_dir
    self.output_dir = args.output_dir
    self.model = face_model.FaceModel(args)

  def _get_identity_directories(self, start_index, end_index):
      self.identity_directory_paths =\
          [self.image_directory_path + '/' + path for path in sorted(os.listdir(self.image_directory_path))][start_index:end_index]

  def _get_image_paths(self):
      self.image_paths =\
          functools.reduce(operator.concat, [[path + '/' + picture_name for picture_name in os.listdir(path)] for path in self.identity_directory_paths])

  def _load_pictures(self):
      self.images = [cv2.imread(path) for path in self.image_paths]

  def _compute_identities(self):
      self.identities = [(path.split('/')[3]).encode("ascii", "ignore") for path in self.image_paths]

  def _calculate_face_vectors(self):
      self.face_vectors = [self.model.get_feature(self.model.get_input(image)) for image in self.images]

  def _write_vectors(self, batch_name):
      file_handle =\
          h5py.File(self.output_dir + '/face_vectors-' + file_number(batch_name, 3) + '.h5', 'w')
      file_handle.create_dataset('face_vectors', data=self.face_vectors)
      file_handle.create_dataset('identities', (len(self.identities),), 'S32', data=self.identities)
      file_handle.close()

  def create_batch_vectors(self, start_index, end_index, batch_number):
      """Create a batch of face vectors based on the start index and end index"""
      self._get_identity_directories(int(start_index), int(end_index))
      self._get_image_paths()
      self._load_pictures()
      self._compute_identities()
      self._calculate_face_vectors()
      self._write_vectors(batch_number)

def run_job(argument_list):
  """Runs the DLib Tester batch job. Called by the Thread pool."""
  embedding_generator = EmbeddingGenerator(argument_list[0])
  embedding_generator.create_batch_vectors(argument_list[1], argument_list[2], argument_list[3])

def main(arguments):
  pool = Pool(processes=int(arguments.processors))
  total_number = len(os.listdir(arguments.image_dir))
  pool_arguments = range_values_and_batches(arguments, total_number)
  # pool.map(run_job, pool_arguments)
  print(total_number)
  run_job([arguments, 0, total_number, 0])

def parse_arguments(argv):
  ap = argparse.ArgumentParser(description='face model test')
  ap.add_argument('--image-size', default='', help='')
  ap.add_argument('--model', default='', help='path to load model.')
  ap.add_argument('--output-dir', default='', help='path to output directory.')
  ap.add_argument('--image-dir', default='', help='path to aligned images.')
  ap.add_argument('--processors', default='', help='number of processors.')
  ap.add_argument('--batch-size', default='', help='batch size.')
  ap.add_argument('--ga-model', default='', help='path to load model.')
  ap.add_argument('--gpu', default=0, type=int, help='gpu id')
  ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
  ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
  ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
  return ap.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
