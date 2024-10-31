export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file=YAML_FILE \
  --log_dir=LOG_DIR \
  --alsologtostderr=True

# # for example

# python egrsdb/main.py \
#   --yaml_file=options/mrsdb-esl-ev-sim-demo.yaml \
#   --log_dir=logs/mrsdb-esl-ev-sim-demo \
#   --alsologtostderr=True

# python egrsdb/main.py \
#   --yaml_file=options/mrsdb-esl-fastec-demo.yaml \
#   --log_dir=logs/mrsdb-esl-fastec-demo \
#   --alsologtostderr=True