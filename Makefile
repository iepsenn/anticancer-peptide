setup:
	conda env create --file environment.yml

train_classifier:
	tacape-train-classifier \
    --positive-train data/raw/anti_cp/anticp2_main_internal_positive.txt \
    --negative-train data/raw/anti_cp/anticp2_main_internal_negative.txt \
    --positive-test data/raw/anti_cp/anticp2_main_validation_positive.txt \
    --negative-test data/raw/anti_cp/anticp2_main_validation_negative.txt \
    --output data/models/classifier \
    --epochs 30

run_classifier:
	tacape-predict \
    --input data/raw/anti_cp/anticp2_main_internal_positive.txt \
    --format text \
    --classifier-prefix data/models/internal \
    --output data/models/internal_results.csv

train_generator:
	tacape-train-generator \
    --positive-train data/raw/anti_cp/anticp2_main_internal_positive.txt \
    --positive-test data/raw/anti_cp/anticp2_main_validation_positive.txt \
    --output data/models/generator \
    --epochs 10

run_generator:
	tacape-generate \
    --generator-prefix data/models/generator \
    --classifier-prefix data/models/classifier \
    --number-of-sequence 100 \
    --output data/models/generated.csv

train: train_classifier train_generator

build_pypi_package:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	python setup.py sdist bdist_wheel

twine_upload: build_pypi_package
	@python setup.py sdist bdist_wheel
	@twine upload \
		--repository-url https://upload.pypi.org/legacy/ \
		-u $(PYPI_USER) \
		-p $(PYPI_PASS) \
		dist/*-py3-none-any.whl