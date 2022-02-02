IMG_SIZE = 480
img_data_shape = (IMG_SIZE, IMG_SIZE, 3)
csv_data_shape = (480, 640)
num_classes = 111

# define two inputs layers
img_input = tf.keras.layers.Input(shape=img_data_shape, name="image")
csv_input = tf.keras.layers.Input(shape=csv_data_shape, name="csv")

# define layers for image data
x1=keras.applications.resnet.ResNet50(include_top=True,
                                   weights='imagenet', input_tensor=None,
                                   input_shape=None, pooling=None, classes=1000)

# define layers for csv data
x2=keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
x2=keras.layers.LSTM(20, return_sequences=True),
x2=keras.layers.TimeDistributed(keras.layers.Dense(10))

# merge layers
x = tf.keras.layers.concatenate([x1,x2], name="concat_csv_img")
x = tf.keras.layers.Dense(128, activation='relu', name="dense1_csv_img")(x)
output = tf.keras.layers.Dense(num_classes, name="classify")(x)

# make model with 2 inputs and 1 output
model = tf.keras.models.Model(inputs=[img_input, csv_input], outputs=output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])