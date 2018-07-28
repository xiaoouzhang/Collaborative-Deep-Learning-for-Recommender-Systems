# Collaborative-Deep-Learning-for-Recommender-Systems
The hybrid model combining stacked denoising autoencoder (SDAE) with matrix factorization (MF) is applied, to predict the customer purchase behavior in the future month according to the purchase history and user information in the [Santander dataset](https://www.kaggle.com/c/santander-product-recommendation). A bolg post for detailed discussions will be available shortly.

This work is contributed by [Sampath Chanda](https://www.linkedin.com/in/sampathchanda/), [Suyin Wang](https://www.linkedin.com/in/suyin-wang-3934b543/) and [Xiaoou Zhang](https://www.linkedin.com/in/xiaoou-zhang-a9559211a/).

The user information is generated in "matrix_factorization.ipynb", and the rating marix is generated in "rating_matrix.py". The main code for the SDAE-MF hybrid model can be found in "mf_auto_mono.py" for one-hidden-layer SDAE and "mf_auto.py" for three-hidden-layer SDAE.

Update: in "mf_auto_mono_v2.py", imporved the speed of matrix factorization taking advantage of the sparsity of the rating matrix.
