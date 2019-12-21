#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

import keras as kr

hello=tf.constant('hello world')

incon=tf.constant(10)
with tf.Session() as sess:
    output=sess.run(hello).decode()
    print(output)
var1=tf.constant('hello')
var2=tf.constant('world')
sum1=var1+" "+var2
with tf.Session() as sess:
    output = sess.run(sum1).decode()
    print(output)

# ## Matrices

r3_matrix = tf.constant([[1,3,5],[4,7,8]])
r3_matrix

matrix_r3 = tf.constant([[1,2,3],
                         [4,5,6],
                         [7,8,9]])
print(matrix_r3)

zerom=tf.zeros(10)
with tf.Session() as sess:
    output = sess.run(zerom)
    print(output)

a = tf.zeros((4,4,3))
b = tf.ones((2,2))
sess= tf.Session()
print(sess.run(a))
print(sess.run(b))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a+b
sess.run(adder,feed_dict={a:[1,2,4],b:[3,6,7]})
# ## Variables

#int k = 1
k = tf.Variable([2.9],dtype=tf.float32)
#whenever we are using Variables need to use tf.global_variables_initializer()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(k))

# ## Linear model

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.9],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w*x+b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4,5]}))
