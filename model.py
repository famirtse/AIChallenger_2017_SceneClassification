import keras
import h5py
import pickle
#from keras.preprocessing.image import ImageDataGenerator
import functools

def main():
    rate=0.75 #dropout rate
    # train_set
    #x_train,y_train=file['x_train'],file['y_train']
    x_train = keras.utils.io_utils.HDF5Matrix('data/train_data.h5', 'x_train')
    y_train = keras.utils.io_utils.HDF5Matrix('data/train_data.h5', 'y_train')
    # val set
    #file1 = h5py.File('data/val_data.h5', 'r')
    #x_val, y_val = file1['x_val'], file1['y_val']
    x_val = keras.utils.io_utils.HDF5Matrix('data/val_data.h5', 'x_val',start=0,end=1000)
    y_val = keras.utils.io_utils.HDF5Matrix('data/val_data.h5', 'y_val',start=0,end=1000)

    base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False)   #调用Xception模型
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)   #池化层
    x = keras.layers.Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l2 (0.001))(x)
    x = keras.layers.Dropout(rate)(x)
    x = keras.layers.Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2 (0.001))(x)
    x = keras.layers.Dropout(rate)(x)   #增加两个全连接层并做L2正则化和两个dropout层
    predictions = keras.layers.Dense(80, activation='softmax')(x)   #输出层
    model =keras.models.Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[:125]:
        layer.trainable = False     #前125层的权重参数从Xception模型迁移，无需训练
    for layer in model.layers[125:]:
        layer.trainable = True

    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy',top3_acc])   #编译模型

    checkpoint = keras.callbacks.ModelCheckpoint('image/model/model1.h5', monitor='val_acc', verbose=1,save_best_only=True, mode='max')
    history=model.fit(x=x_train,y=y_train,batch_size=64,epochs=20,validation_data=(x_val,y_val),callbacks=[checkpoint],shuffle='batch')   #训练模型

    with open('data/history.pkl', 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)   #将训练结果序列化
if __name__ == "__main__":
    main()