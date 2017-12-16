#generate rating matrix

import numpy as np
import pandas as pd

store = pd.HDFStore('ratingDF_tr.h5')
df_rating = store['df_rating_tr']

user_id=np.array(df_rating['ncodpers'].unique())
prod_id=np.array(df_rating['prodIdx'].unique())
prod_id=np.sort(prod_id)
#u_dictionary=dict(zip(user_id,np.arange(user_id.size)))

rating_mat=pd.DataFrame(np.zeros((user_id.size,prod_id.size)),user_id,prod_id)
for i in range(df_rating.shape[0]):
    #sample_prod_id=df_rating.iloc[i].prodIdx
    #rating_mat.loc[df_rating.iloc[i].ncodpers][rating_mat.columns[df_rating.iloc[i].prodIdx-1]]=df_rating.iloc[i].rating
    rating_mat.loc[df_rating.iloc[i].ncodpers,df_rating.iloc[i].prodIdx]=df_rating.iloc[i].rating

store = pd.HDFStore('rating_mat_tr.h5')
store['rating_mat'] = rating_mat
store.close()

print('training set complete')

store = pd.HDFStore('ratingDF_val.h5')
df_rating = store['df_rating_val']

#u_dictionary=dict(zip(user_id,np.arange(user_id.size)))

rating_mat=pd.DataFrame(np.zeros((user_id.size,prod_id.size)),user_id,prod_id)
for i in range(df_rating.shape[0]):
    if df_rating.iloc[i].ncodpers in rating_mat.index:
        #print('old')
        #rating_mat.loc[df_rating.iloc[i].ncodpers,df_rating.iloc[i].prodIdx]=rating_mat.loc[df_rating.iloc[i].ncodpers,df_rating.iloc[i].prodIdx]+df_rating.iloc[i].rating
        rating_mat.loc[df_rating.iloc[i].ncodpers,df_rating.iloc[i].prodIdx]=df_rating.iloc[i].rating
    else:
        print('new')
    print(i)

store = pd.HDFStore('rating_mat_val.h5')
store['rating_mat'] = rating_mat
store.close()
