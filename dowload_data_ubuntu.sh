pip install kaggle
kaggle datasets download shaunthesheep/microsoft-catsvsdogs-dataset
mkdir data
unzip microsoft-catsvsdogs-dataset.zip -d data
mv data/PetImages data/train
