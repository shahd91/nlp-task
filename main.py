import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# تحميل الداتا سيت
file_path = r'C:\Users\shahd\.cache\kagglehub\datasets\d4rklucif3r\restaurant-reviews\versions\1\Restaurant_Reviews.tsv'
df = pd.read_csv(file_path, sep='\t')

# معالجة البيانات
df = df[['Review', 'Liked']]
df.dropna(inplace=True)
df['Liked'] = LabelEncoder().fit_transform(df['Liked'])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Liked'], test_size=0.2, random_state=42)

# تحويل النصوص إلى تسلسل رقمي
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# ضبط طول التسلسل
maxlen = 100
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

# بناء النموذج
model = tf.keras.Sequential([
 tf.keras.layers.Embedding(input_dim=5000, output_dim=128),
 tf.keras.layers.SimpleRNN(128),
 tf.keras.layers.Dense(1, activation='sigmoid')
])

# تجميع النموذج
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test))

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')