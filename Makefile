DATASET := rival_identities
MODEL := model-r50-am-lfw
IMAGE_SIZE := 112

TRAINING_DATASET := faces_vgg_112x112
TRAINING_MODEL := model-r100

generate_embeddings:
	mkdir -p computed_data/${DATASET}/${MODEL}
	python rival-src/embedding_generator.py --image-size "${IMAGE_SIZE},${IMAGE_SIZE}" --model "./models/${MODEL}/model,0" --image-dir "./aligned_datasets/${DATASET}_mtcnnpy_${IMAGE_SIZE}" --output-dir computed_data/${DATASET}/${MODEL} --processors 4 --batch-size 4000

compare:
	python rival-src/comparison.py --model ${MODEL} --dataset ${DATASET}

generate_aligned_dataset:
	export PYTHONPATH=/home/powerbear/facenet/src && for N in {1..4}; do python src/align/align_dataset_mtcnn.py datasets/${DATASET}/ aligned_datasets/${DATASET}_mtcnnpy_${IMAGE_SIZE}/ --image_size ${IMAGE_SIZE} --margin 32 --random_order --gpu_memory_fraction 0.25 & done

complete_comparison: generate_embeddings compare

train_model:
	CUDA_VISIBLE_DEVICES='0,1,2,3' python -u src/train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir datasets/${TRAINING_DATASET} --prefix models/${TRAINING_MODEL}
