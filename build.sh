#! /bin/bash
set -e

: ${BUILD_PATH:=build}

python3.8 -m venv /tmp/env
source /tmp/env/bin/activate

# Build documentation
pip install -r docs/requirements.txt
doc8 --max-line-length 130 docs/
make checklinks
make html
# exit when asked to build doc
if [[ "$1" == "build_doc" ]]
then
  exit 0  
fi


pip install ruff
# remove --exit-zero once all errors are fixed/explicitly ignore
ruff check --line-length=120 --ignore=F401,E203
# exit when asked to run `ruff` only
if [[ "$1" == "ruff" ]]
then
  exit 0
fi

pip install wheel
python setup.py bdist_wheel --dist-dir ${BUILD_PATH}/pip/public/neuronx-distributed-training
