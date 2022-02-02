#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve, train_test_split
import json
import keras
#%%
path='train2/train'
#
number =os.listdir(path)

#데이터 일단 10개씩만 불러옴...ㅠㅠ
# sample_csv10=[pd.read_csv((path+'/{}'.format(number[i])+'/{}.csv'.format(number[i])).replace("\\",'/')) for i in range(0,10)]
# sample_image = [cv2.imread((path+'/{}'.format(number[i])+'/{}.jpg'.format(number[i])).replace("\\",'/')) for i in range(0,10)]
# sample_json = []

for i in range(0,10):
    with open((path+'/{}'.format(number[i])+'/{}.json'.format(number[i])).replace("\\",'/')) as f:
        sample_json.append(json.load(f))
#%%
#데이터를 둘러보자~!
plt.imshow(cv2.cvtColor(sample_image[1], cv2.COLOR_BGR2RGB))
plt.show()

#bbox 그게 몬데...? 여튼 보이게 해준다네요

# visualize bbox
plt.figure(figsize=(7,7))
points = sample_json[0]['annotations']['bbox'][0]
part_points = sample_json[0]['annotations']['part']
img = cv2.cvtColor(sample_image[0], cv2.COLOR_BGR2RGB)

cv2.rectangle(
    img,
    (int(points['x']), int(points['y'])),
    (int((points['x']+points['w'])), int((points['y']+points['h']))),
    (0, 255, 0),
    2
)
for part_point in part_points:
    point = part_point
    cv2.rectangle(
        img,
        (int(point['x']), int(point['y'])),
        (int((point['x']+point['w'])), int((point['y']+point['h']))),
        (255, 0, 0),
        1
    )
plt.imshow(img)
plt.show()
#%%
#환경데이터 통계량을 계산하자고 하네요 응 다베꼇어

# 분석에 사용할 feature 선택
csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

csv_files = sorted(glob('train2/train/*/*.csv'))

temp_csv = pd.read_csv(csv_files[0])[csv_features]
max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

# feature 별 최대값, 최솟값 계산 #max_arr 과 min_arr을 계속 갱신하는 거 같음
for csv in tqdm(csv_files[1:]):
    temp_csv = pd.read_csv(csv)[csv_features]
    temp_csv = temp_csv.replace('-',np.nan).dropna()
    if len(temp_csv) == 0:
        continue
    temp_csv = temp_csv.astype(float)
    temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
    max_arr = np.max([max_arr,temp_max], axis=0)
    min_arr = np.min([min_arr,temp_min], axis=0)

# feature 별 최대값, 최솟값 dictionary 생성
csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}

# {'내부 온도 1 평균': [3.4, 47.3],
#  '내부 온도 1 최고': [3.4, 47.6],
#  '내부 온도 1 최저': [3.3, 47.0],
#  '내부 습도 1 평균': [23.7, 100.0],
#  '내부 습도 1 최고': [25.9, 100.0],
#  '내부 습도 1 최저': [0.0, 100.0],
#  '내부 이슬점 평균': [0.1, 34.5],
#  '내부 이슬점 최고': [0.2, 34.7],
#  '내부 이슬점 최저': [0.0, 34.4]}

# CustomDataset 제작 #올것이왔다 #이걸내가 이해할수있을까 #결국못하고고침


# 변수 설명 csv 파일 참조
# crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
# disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
#            '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
# risk = {'1':'초기','2':'중기','3':'말기'}

label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
list(label_description.items())[:10]

label_encoder = {key:idx for idx, key in enumerate(label_description)}
label_decoder = {val:key for key, val in label_encoder.items()}
#%%
################개발구역 조심조심####################
#csv

def csv_trans(path,file_name):
    csv_path=f'{path}/{file_name}/{file_name}.csv'
    df=pd.read_csv(csv_path)[csv_feature_dict.keys()]
    df = df.replace('-', 0)
    # MinMax scaling
    for col in df.columns:
        for col in df.columns:
            df[col] = df[col].astype(float) - csv_feature_dict[col][0]
            df[col] = df[col] / (csv_feature_dict[col][1] - csv_feature_dict[col][0])

    # zero padding
    pad = np.zeros((max_len, len(df.columns)))
    length = min(max_len, len(df))
    pad[:length] = df.to_numpy()[:length]

    # transpose to sequential data
    csv_feature = pad.T
    return csv_feature
#%%
#jpg
def image_trans(path,file_name):

    jpg_path=f'{path}/{file_name}/{file_name}.jpg'
    img = cv2.imread(jpg_path)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))

    return img
#####################
csv_features=[]
imgs=[]
for i in tqdm(range(5767)):
    csv_features.append(csv_trans(path, number[i]))

csv_features=np.array(csv_features)
csv_features.shape
#
for i in tqdm(range(5767)):
    imgs.append(image_trans(path,number[i]))

imgs=np.array(imgs)
imgs.shape
#%%
# batch_size = 256
# class_n = len(label_encoder)
# learning_rate = 1e-4
# embedding_dim = 512
# num_features = len(csv_feature_dict)
# max_len = 24*6
# dropout_rate = 0.1
# epochs = 10
# vision_pretrain = True
# save_path = 'best_model.h5'
#%%
#이해는 다음에 하는 것으루 하자 따흐흑바흐흑 #아니다 이해중

#모델링을 하자!!! 으아아아아아악!!!!!!
IMG_SIZE = 224

num_crop = 6
num_disease = 35
num_risk = 4
#%%
# define two inputs layers
img_input = tf.keras.layers.Input(shape=[224,224,3], name="image")
csv_input = tf.keras.layers.Input(shape=[9,144], name="csv")

# define layers for image data
x1=keras.applications.resnet.ResNet50(include_top=False,
                                   weights='imagenet',
                                   input_shape=[224,224,3])
x1.trainable=False
x1_tensor=x1(img_input*255)
x1_tensor=tf.reshape(x1_tensor,shape=[-1,49,2048],name='reshape')
x1_tensor=tf.keras.layers.Dense(32)(x1_tensor)

# define layers for csv data
x2_tensor=keras.layers.LSTM(20, return_sequences=True)(csv_input)
x2_tensor=keras.layers.LSTM(20, return_sequences=True)(x2_tensor)
x2_tensor=keras.layers.TimeDistributed(keras.layers.Dense(32))(x2_tensor)

# merge layers
x = tf.keras.layers.concatenate([x1_tensor,x2_tensor],axis=1, name="concat_csv_img")
x = tf.keras.layers.Flatten(name="flatten_csv")(x)

crop_pred = tf.keras.layers.Dense(num_crop,activation='softmax', name="crop_pred")(x)
disease_pred = tf.keras.layers.Dense(num_disease,activation='softmax', name="disease_pred")(x)
risk_pred=tf.keras.layers.Dense(num_risk,activation='softmax',name="risk_pred")(x)

model=keras.Model(
    inputs=[img_input,csv_input],
    outputs=[crop_pred,disease_pred,risk_pred],
)

# make model with 2 inputs and 1 output

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=[keras.losses.SparseCategoricalCrossentropy(),
                    keras.losses.SparseCategoricalCrossentropy(),
                    keras.losses.SparseCategoricalCrossentropy()]
              )
#%%
#############################
#train
img_data=imgs
csv_data=csv_features

#y cleaning
disease_dict={'0': 0,
              'a1': 1,'a2': 2,'a3':3,'a4':4,
              'a5':5,'a6':6,'a7':7,'a8':8,'a9':9,'a10':10,
              'a11':11,'a12':12,
              'b1': 13,'b2':14,'b3':15,'b4':16,'b5':17,'b6':18,'b7':19,'b8':20}

disease_dict_reverse={v:k for k,v in disease_dict.items()}

def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]


crop_targets=np.array(pd.read_csv('train.csv')['crop']) - 1 #0부터 시작하게
disease_targets=np.array(replace_with_dict(np.array(pd.read_csv('train.csv')['disease']),disease_dict))
risk_targets=np.array(pd.read_csv('train.csv')['risk'])

#convert output to tensor
crop_targets=tf.convert_to_tensor(crop_targets,dtype=tf.float32)
disease_targets=tf.convert_to_tensor(disease_targets,dtype=tf.float32)
risk_targets=tf.convert_to_tensor(risk_targets,dtype=tf.float32)

#convert input to tensor
img_data=tf.convert_to_tensor(np.asarray(imgs),dtype=tf.float32)
img_data=tf.reshape(img_data,shape=[5767,224,224,3])

csv_data=tf.convert_to_tensor(np.asarray(csv_features),dtype=tf.float32)
#%%
model.fit(
    [img_data,csv_data],
    [crop_targets ,disease_targets ,risk_targets],
    epochs=100,
    batch_size=256,
    validation_split=0.2
)
#%%
#test

# del imgs
# del csv_features

#메모리 오류났다.. 일단 csv 랑 model 저장하자
model.save('220201model.h5')#모델저장
model = tf.keras.models.load_model('220201model.h5')
#%%

path_test='test/test'
number_test =os.listdir(path_test)

csv_features_test=[]
imgs_test=[]

for i in tqdm(range(51906)):
    csv_features_test.append(csv_trans(path_test, number_test[i]))

csv_features_test=np.array(csv_features_test)
csv_features_test.shape

#

#
for i in tqdm(range(51906)):
    imgs_test.append(image_trans(path_test,number_test[i]))

imgs_test=np.array(imgs_test)
imgs_test.shape

# img_data=tf.convert_to_tensor(np.asarray(imgs_test),dtype=tf.float32)
# img_data=tf.reshape(img_data,shape=[1,224,224,3])
#
# csv_data=tf.convert_to_tensor(np.asarray(csv_features_test),dtype=tf.float32)
k=0
# 51906
result_df = pd.DataFrame(index=range(1), columns=['pred_crop','pred_disease','pred_risk'])
for k in tqdm(range(51906)):

    img_data = tf.convert_to_tensor(np.asarray(imgs_test[k]), dtype=tf.float32)
    img_data = tf.reshape(img_data, shape=[1, 224, 224, 3])

    csv_data = tf.convert_to_tensor(np.asarray(csv_features_test[k]), dtype=tf.float32)
    csv_data = tf.reshape(csv_data, shape=[1, 9, 144])

    pred_prob_=model.predict([img_data,csv_data])

    argmaxlist=[]
    for i in range(3):
        b=[]
        for j in range(len(pred_prob_[i])):
            a=np.argmax(pred_prob_[i][j])
            b.append(a)
        argmaxlist.append(b)

    argmax_df=pd.DataFrame(
        {'pred_crop':argmaxlist[0],
         'pred_disease':argmaxlist[1],
         'pred_risk':argmaxlist[2]}
    )

    #다시 원래대로 바꾸는 작업
    argmax_df['pred_crop']=argmax_df['pred_crop']+1
    argmax_df['pred_disease']=argmax_df['pred_disease'].apply(lambda x:
                                                              disease_dict_reverse.get(x))
    result_df=pd.concat([result_df,argmax_df])
#세개를 잇고 익스포트
#아니다 그건 엑셀로 하자 ㅠ
result_df.to_csv("pred_label220201.csv")
#%%