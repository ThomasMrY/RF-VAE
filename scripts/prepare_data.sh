#! /bin/sh
mkdir -p data
cd data
git clone https://github.com/deepmind/dsprites-dataset.git
cd data/dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5

