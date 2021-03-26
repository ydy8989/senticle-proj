import re
from collections import Counter
from functools import partial

import numpy as np
import tensorflow as tf
from keras import backend as K
from livelossplot import PlotLossesKeras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tokenization_kobert import KoBertTokenizer
from tqdm import tqdm
from transformers import *
from preprocessing import *
if __name__ == '__main__':

    #
    # train = df#.iloc#[:int(len(df)*0.8)]
    # # test =  df.iloc[int(len(df)*0.8):]
    # train['tok'] = train['text'].apply(lambda x: re.sub('[^가-힣a-zA-Z]', ' ', x))
    # # test['tok'] = test['text'].apply(lambda x: re.sub('[^가-힣a-zA-Z]',' ',x))
    # # test = test.reset_index(drop=True)
    # tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    #
    # X_train_words = []
    # for sentence in train['tok']:
    #     temp_X = []
    #     temp_X = tokenizer.tokenize(sentence)
    #     X_train_words.append(temp_X)
    #
    # # X_test_words = []
    # # for sentence in test['tok']:
    # #     temp_X = []
    # #     temp_X = tokenizer.tokenize(sentence)
    # #     X_test_words.append(temp_X)
    # train['tokenized'] = X_train_words
    # # test['tokenized'] = X_test_words
    #
    # # 레이블별 token의 등장 횟수 in train set
    #
    #
    # label_stack = [Counter(np.hstack(train[train.label == i]['tokenized'].values)) for i in range(2)]
    # # label_stack
    # for idx, i in enumerate(label_stack):
    #     print('label_{}:'.format(idx), len(train[train.label == idx]), '건 =>', i.most_common(10))
    #
    #
    # def data_convert(data_df):
    #     global tokenizer
    #     # 버트에 들어갈 인풋의 길이
    #     global SEQ_LEN
    #
    #     tokens, masks, segments, targets = [], [], [], []
    #     for i in tqdm(range(len(data_df))):
    #         # Tokenizing 진행 및 남는 길이만큼 패딩
    #         token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True)
    #         token = token + [0] * (SEQ_LEN - len(token))
    #
    #         # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
    #         num_zeros = token.count(0)
    #         mask = [1] * (SEQ_LEN - num_zeros) + [0] * num_zeros
    #
    #         # 우리의 경우 문장이 1개밖에 없으므로 모두 0
    #         segment = [0] * SEQ_LEN
    #
    #         # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
    #         tokens.append(token)
    #         masks.append(mask)
    #         segments.append(segment)
    #
    #         # targets 변수에 label을 저장
    #         targets.append(data_df[LABEL_COLUMN][i])
    #
    #     # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정
    #     tokens = np.array(tokens)
    #     masks = np.array(masks)
    #     segments = np.array(segments)
    #     targets = np.array(targets)
    #
    #     return [tokens, masks, segments], targets
    #
    #
    # # 위에 정의한 data_convert 함수를 불러오는 함수를 정의
    # def load_data(pandas_dataframe):
    #     data_df = pandas_dataframe
    #     data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    #     data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    #     print(data_df[LABEL_COLUMN])
    #     data_x, data_y = data_convert(data_df)
    #
    #     return data_x, data_y
    #
    #
    # # train, test 데이터 모두 cover 가능한 사이즈의 SEQ_LEN로 지정
    # # 앞서 확인한 토큰의 가장 긴 길이보다 큰 값으로 인풋의 길이를 정해준다.
    # SEQ_LEN = 512
    #
    # # 데이터프레임의 데이터 부분과 레이블 부분의 column명 명시
    # DATA_COLUMN = "tok"
    # LABEL_COLUMN = "label"
    #
    # # train 데이터를 버트 인풋에 맞게 변환
    # train_x, train_y = load_data(train)
    #
    # # test 데이터를 버트 인풋에 맞게 변환
    # # test_x, test_y = load_data(test)
    # # shape 확인.
    # print(train_x[0].shape, train_x[1].shape, train_y.shape)
    # # print(test_x[0].shape, test_x[1].shape, test_y.shape)
    #
    #
    # # 분류를 위한 클래스 one-hot-encoding
    # encoder = LabelEncoder()
    # encoder.fit(train_y)
    # encoded_train_y = to_categorical(encoder.transform(train_y))
    # # encoded_test_y = to_categorical(encoder.transform(test_y))
    #
    #
    # top3_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=3)
    # top5_acc = partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)
    # top3_acc.__name__ = 'top3_acc'
    # top5_acc.__name__ = 'top5_acc'
    #
    #
    #
    # def recall_metric(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     return recall
    #
    #
    # def precision_metric(y_true, y_pred):
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     return precision
    #
    #
    # def f1_score(y_true, y_pred):
    #     precision = precision_metric(y_true, y_pred)
    #     recall = recall_metric(y_true, y_pred)
    #     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    #
    #
    #
    # def create_classification_bert():
    #     # 버트 pretrained 모델 로드
    #     # model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
    #     model = TFBertModel.from_pretrained('monologg/kobert', from_pt=True)
    #     # print(model.summary())
    #     # 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
    #     token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
    #     mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
    #     segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
    #
    #     # 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
    #     bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
    #     print(bert_outputs)
    #
    #     # 분류 클래스 17
    #     bert_outputs = bert_outputs[1]
    #     classification_first = tf.keras.layers.Dense(2, activation='softmax',
    #                                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(
    #                                                      stddev=0.02))(bert_outputs)
    #     classification_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], classification_first)
    #     classification_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.0e-5),
    #                                  loss=tf.keras.losses.CategoricalCrossentropy(),
    #                                  metrics=['accuracy', f1_score])  # top3_acc, top5_acc,
    #     print(classification_model.summary())
    #     return classification_model
    # # pretrained_model = create_classification_bert()
    #
    #
    # # validation loss가 5번 증가하면 종료합니다.
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    # # GPU 모드로 훈련시킬 때
    # classification_model = create_classification_bert()
    #
    # classification_model.fit(train_x,
    #                          encoded_train_y,
    #                          epochs=50,
    #                          # shuffle=True,
    #                          batch_size=2,
    #                          validation_split=0.2,
    #                          callbacks=[es, PlotLossesKeras()])
    # #                        validation_data=(test_x, encoded_test_y))
    #
    #
    #
