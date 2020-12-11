import tensorflow as tf
from tensorflow.keras import datasets,layers,Sequential,losses,optimizers
import matplotlib.pyplot as plt

#动态分配显存
physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

#
# 没什么用
# 还是显存不足。。。
#

#超参数
shuffleSize = 60000
batchSize = 60000
lr = tf.constant(1e-4,dtype=tf.float32) #训练步长
lossList = [] #损失可视化
criteon = losses.CategoricalCrossentropy(from_logits=True) #交叉熵损失函数

#准备数据
(x,y), (x_test,y_test) = datasets.mnist.load_data()

x = tf.constant(x, dtype=tf.float32)        #(60000, 28, 28)
y = tf.constant(y, dtype = tf.int32)        #(60000,)
x = tf.expand_dims(x, axis=3)
y_onehot = tf.one_hot(y, depth=10)
train_set = tf.data.Dataset.from_tensor_slices((x,y_onehot)).shuffle(shuffleSize).batch(batchSize)

x_test = tf.constant(x_test, dtype=tf.float32)#(10000, 28, 28)
y_test = tf.constant(y_test, dtype = tf.int32)#(10000,)
x_test = tf.expand_dims(x_test, axis=3)
y_test_onehot = tf.one_hot(y_test, depth=10)
test_set = tf.data.Dataset.from_tensor_slices((x_test,y_test_onehot)).shuffle(shuffleSize).batch(batchSize)

#构建网络
network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),
    layers.MaxPooling2D(pool_size=2, strides=2),#高宽减半的池化层
    layers.ReLU(),

    layers.Conv2D(16, kernel_size=3, strides=1),
    layers.MaxPooling2D(pool_size=2, strides=2),#高宽减半的池化层
    layers.ReLU(),
    layers.Flatten(),

    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])
# network.build(input_shape=(4,28,28,1))
# print(network.summary())

#训练模型
for Epoch in range(30):
    with tf.GradientTape() as tape:
        out = network(x)
        loss =  criteon(y_onehot, out)
        print('loss:',loss)
    lossList.append(float(loss.numpy()))
    grad = tape.gradient(loss, network.trainable_variables)
    zipped = zip(grad, network.trainable_variables)
    for ele in zipped:
        ele[1].assign_sub(lr*ele[0])

out1 = tf.constant([])
tag = 0
for a,b in train_set:
    tag+=1
    if tag ==1:
        out1 = tf.concat(network(a), axis=0)
    else:
        out1 = tf.concat([out1,network(a)], axis=0)
pred1 = tf.cast(tf.argmax(out1, axis=-1), dtype=tf.int32)
acc1 = tf.reduce_sum(tf.cast(tf.equal(pred1, y), dtype=tf.float32)) / float(x.shape[0])

out2 = network(x_test)
pred2 = tf.cast(tf.argmax(out2, axis=-1), dtype=tf.int32)
acc2 = tf.reduce_sum(tf.cast(tf.equal(pred2, y_test), dtype=tf.float32)) / float(x.shape[0])

print('trian accuracy:',str(float(acc1)*100)+'%')
print('test accuracy:',str(float(acc2)*100)+'%')

fig,ax = plt.subplots()
ax.plot(lossList,'-b.')
plt.show()
