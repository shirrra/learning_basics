# basics of TF - based on tut. https://www.tensorflow.org/get_started/get_started
import tensorflow as tf
# creating two floating point tensorts
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
# printing. this does not evaluate the variables
print(node1, node2)
# create a session object
sess = tf.Session()
# run and print the session on our nodes, this will evaluate them
print(sess.run([node1, node2]))
# more complicated example
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
# using placeholders instead of constants
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # provides a shortcut for tf
print(sess.run(a, {a:5}))
print(sess.run(adder_node, {a:5, b:6}))
print(sess.run(adder_node, {a:[5,-2], b:[6,2]}))
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:9, b:2}))
# adding variables, which shall be used as weights
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# initializing
init = tf.global_variables_initializer()
sess.run(init)
# evaluating our linear_model
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# evaluate how good is our linear_model
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # gives 23.66
# fixing our parameters to the best ones
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # gives 0.0
# learning the best parameters
optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)
# reset our weights to the original (bad) values
sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# print which values we got (for the weights)
print(sess.run([W, b]))
# print the current loss
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # gives 23.66
