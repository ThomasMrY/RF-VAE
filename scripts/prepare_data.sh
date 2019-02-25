#! /bin/sh
mkdir -p data
cd data
git clone https://github.com/deepmind/dsprites-dataset.git
ls -a
cd data/dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5