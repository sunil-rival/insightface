DATASET := lfw
MODEL := model-r50-am-lfw
IMAGE_SIZE := 112

generate_embeddings:
	mkdir -p computed_data/${DATASET}/${MODEL}
	python rival-src/embedding_generator.py --image-size "${IMAGE_SIZE},${IMAGE_SIZE}" --model "./models/${MODEL}/model,0" --image-dir "./aligned_datasets/${DATASET}_mtcnnpy_${IMAGE_SIZE}" --output-dir computed_data/${DATASET}/${MODEL} --processors 4 --batch-size 4000

compare:
	python rival-src/comparison.py --model ${MODEL} --dataset ${DATASET}

generate_aligned_dataset:
	export PYTHONPATH=/home/powerbear/facenet/src && for N in {1..4}; do python src/align/align_dataset_mtcnn.py datasets/${DATASET}/ aligned_datasets/${DATASET}_mtcnnpy_${IMAGE_SIZE}/ --image_size ${IMAGE_SIZE} --margin 32 --random_order --gpu_memory_fraction 0.25 & done

complete_comparison: generate_embeddings compare

train_model:
	python src/train_tripletloss.py --logs_base_dir logs/ --models_base_dir models/facenet --data_dir aligned_datasets/${DATASET}_mtcnnpy_${IMAGE_SIZE} --people_per_batch 9 --embedding_size 512 --image_size ${IMAGE_SIZE} --gpu_memory_fraction 0.75 --max_nrof_epochs 100
