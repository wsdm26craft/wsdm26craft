#!/bin/sh

pip uninstall pyqlib
cd qlib_source
pip install .
cd ..
cd master
for seed in 0 1 2 3 4
do
    python main.py $1 $2 $seed
done
cd ..
cd matcc
for seed in 0 1 2 3 4
do
    python main.py $1 $2 $seed
done
cd ..
cd ..
for seed in 0 1 2 3 4
do
  python main.py --dataset="../dataset/$2_$1.pkl" --model_save_path="../model/$2_$1_model.pkl"
  python test.py --dataset="../dataset/$2_$1.pkl" --model_path="../model/$2_$1_model.pkl" --result_save_path="eval/result/$2_$1_result.pkl"
done
echo "all results has been saved in eval/result."