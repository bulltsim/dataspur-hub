# Makefile for DataSpur Phase 1

.PHONY: setup lfs detect actions frames submodules sparse

setup:
	pip install -r requirements.txt

lfs:
	git lfs install

detect:
	bash detect/train_yolo.sh

actions:
	python action/train_c3d.py --data_root datasets/frames

frames:
	bash action/utils/extract_frames.sh

submodules:
	git submodule update --init --recursive

sparse:
	(cd content/pbr-blog && git config core.sparseCheckout true && git sparse-checkout init --cone && git sparse-checkout set config && git pull || true)
	(cd datasets/assets-capstone-img && git config core.sparseCheckout true && git sparse-checkout init --cone && git sparse-checkout set Capstone-img && git pull || true)
