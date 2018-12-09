from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

# GPU参数训练设置，CPU可注释掉
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
config.gpu_options.allow_growth = True

# 数据集地址
path = "./fruits/"
# 模型保存地址
model_path = "./module/model"
# 记录保存地址（用于绘制神经网络结构）
log_path = "./log"

# 设置网络输入图片的格式为100 * 100 * 3
w = 100
h = 100
c = 3


# 读取数据集
def read_image(path):

    global w
    global h
    global c
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            images.append(img)
            labels.append(index)
    return np.asarray(images, np.float32), np.asarray(labels, np.int32)


data, label = read_image(path)
# 使数据集乱序
num_example = data.shape[0]
array = np.arange(num_example)
np.random.shuffle(array)
data = data[array]
label = label[array]
# 训练集: 验证集 = 7: 3, 考虑到样本较少，验证集的结果可以反映测试集结果
sample = np.int(num_example * 0.7)
x_train = data[: sample]
y_train = label[: sample]
x_val = data[sample:]
y_val = label[sample:]


# 绘制参数变化
def variable_summaries(var):

    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


# 构建LeNet-5卷积神经网络
def inference(input_tensor, train, regularizer):

    # 第一层卷积层，输入100 * 100 * 3，输出100 * 100 * 64
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 3, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(conv1_weights)
        conv1_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        variable_summaries(conv1_biases)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        tf.summary.histogram('relu1', relu1)

    # 第二层池化层，输入100 * 100 * 64，输出50 * 50 * 64
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        tf.summary.histogram('pool1', pool1)

    # 第三层卷积层，输入50 * 50 * 64，输出50 * 50 * 128
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(conv2_weights)
        conv2_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        variable_summaries(conv2_biases)
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        tf.summary.histogram('relu2', relu2)

    # 第四层卷积层，输入50 * 50 * 128，输出25 * 25 * 128，并进行扁平化处理
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 25 * 25 * 128
        reshaped = tf.reshape(pool2, [-1, nodes])
        tf.summary.histogram('pool2', pool2)

    # 第五层全连接层，隐含节点为1024个
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc1_weights)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        variable_summaries(fc1_biases)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
        tf.summary.histogram('fc1', fc1)

    # 第六层全连接层，隐含节点为512个
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc2_weights)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        variable_summaries(fc2_biases)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
        tf.summary.histogram('fc2', fc2)

    # 第七层全连接层，输出3种分类
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 3], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc3_weights)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [3], initializer=tf.constant_initializer(0.1))
        variable_summaries(fc3_biases)
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
        tf.summary.histogram('logit', logit)

    return logit


# 占位符，设置输入参数的大小和格式
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y = tf.placeholder(tf.int32, shape=[None, ], name='y')
# 设置正则化参数为0.0001
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, True, regularizer)
b = tf.constant(value=1, dtype=tf.float32)
logits_tag = tf.multiply(logits, b, name='logits_tag')
# 设置损失函数
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
# 使用AdamOptimizer（自适应梯度优化算法）优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# 训练块数据生成器
def get_feed_data(inputs=None, targets=None, batch_size=None, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 初始会话并开始训练过程
epoch = 50
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path, sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    # 训练
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in get_feed_data(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([optimizer, cross_entropy, accuracy], feed_dict={x: x_train_a, y: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("Epoch: %02d" % (i + 1) + " train loss: %f, train accuracy: %f" % (np.sum(train_loss) / n_batch, np.sum(train_acc) / n_batch))

    # 验证
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in get_feed_data(x_val, y_val, batch_size, shuffle=False):
        summary, err, ac = sess.run([merged, cross_entropy, accuracy], feed_dict={x: x_val_a, y: y_val_a})
        val_loss += err
        val_acc += ac
        writer.add_summary(summary, i)
        n_batch += 1
    print("Epoch: %02d" % (i + 1) + " validation loss: %f, validation accuracy: %f" % (np.sum(val_loss) / n_batch, np.sum(val_acc) / n_batch))

print("training finished.")
saver.save(sess, model_path)
writer.close()
sess.close()
