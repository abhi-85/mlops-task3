#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[ ]:


from keras.layers import MaxPooling2D


# In[ ]:


from keras.layers import Flatten


# In[ ]:


from keras.layers import Dense


# In[ ]:


from keras.models import Sequential


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Flatten())


# In[ ]:


model.add(Dense(units=128, activation='relu'))


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(units=1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=1000,
        epochs=25,
        validation_data=test_set,
        validation_steps=700)

