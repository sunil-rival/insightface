import numpy as np
from numpy.linalg import norm
import itertools
import heapq
import argparse
import h5py

def calculate_greatest_intra_identity_distance(identity):
  embedding_vectors = embedding_vectors_for_identity(identity)
  if (len(embedding_vectors) > 1):
    return max_of_pairwise_distances(embedding_vectors)
  else:
    return 0.0

def find_identity_indexes(identity):
  return np.where(labels == identity)[0]

def embedding_vectors_for_identity(identity):
  return embeddings[find_identity_indexes(identity)]

def max_of_pairwise_distances(vectors):
  return max([norm(pair[0]-pair[1]) for pair in list(itertools.combinations(vectors, 2))])

def index_of_closest_vector_for_index(index):
  norms = [norm(embeddings[index]-value) for value in embeddings]
  second_smallest_norm = heapq.nsmallest(2, norms)[-1]
  if (len(find_identity_indexes(labels[index])) > 1):
    return_value = norms.index(second_smallest_norm)
  else:
    return_value = -1
  return return_value

def identity_of_closest_vector_for_index(index):
  global num_correct
  global num_incorrect
  index_value = index_of_closest_vector_for_index(index) 
  if (index_value == -1):
    return -1
  else:
    return_value = labels[index_of_closest_vector_for_index(index)]
    if (return_value != labels[index]):
      num_incorrect += 1
      print('Incorrect identity: (Total correct - ' + str(num_correct) + ' & Total incorrect - ' + str(num_incorrect) + ')')
    else:
      num_correct += 1
      print('Correct identity: (Total correct - ' + str(num_correct) + ' & Total incorrect - ' + str(num_incorrect) + ')')
    return return_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="The name of the model being used.")
    parser.add_argument('--dataset', type=str, help="The name of the dataset.")

    args = parser.parse_args()

    root_string = './computed_data/' + args.dataset + '/' + args.model + '/face_vectors-001.h5'
    file_handle = h5py.File(root_string)
    embeddings = np.array(file_handle.get('face_vectors'))
    labels = np.array(file_handle.get('identities'))
    max_identity = len(set(labels))
    num_correct = 0
    num_incorrect = 0

    # greatest_intra_identity_distances = [calculate_greatest_intra_identity_distance(identity) for identity in np.arange(max_identity)]
    # greatest_intra_identity_distances.sort()

    identity_of_closest_vectors_for_embeddings = np.array([identity_of_closest_vector_for_index(index) for index in np.arange(len(embeddings))])
    print(identity_of_closest_vectors_for_embeddings)

