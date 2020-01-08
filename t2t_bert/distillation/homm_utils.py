import tensorflow as tf

def HoMM(xs, xt, order=3, num=300000):
	xs = xs - tf.reduce_mean(xs, axis=0)
	xt = xt - tf.reduce_mean(xt, axis=0)
	dim = tf.cast(xs.shape[1], tf.int32)
	index = tf.random_uniform(shape=(num, dim), minval=0, maxval=dim - 1, dtype=tf.int32)
	index = index[:, :order]
	xs = tf.transpose(xs)
	xs = tf.gather(xs, index)  ##dim=[num,order,batchsize]
	xt = tf.transpose(xt)
	xt = tf.gather(xt, index)
	HO_Xs = tf.reduce_prod(xs, axis=1)
	HO_Xs = tf.reduce_mean(HO_Xs, axis=1)
	HO_Xt = tf.reduce_prod(xt, axis=1)
	HO_Xt = tf.reduce_mean(HO_Xt, axis=1)
	return tf.reduce_mean(tf.square(tf.subtract(HO_Xs, HO_Xt)))

def HoMM3_loss(xs, xt):
	xs = xs - tf.reduce_mean(xs, axis=0)
	xt = xt - tf.reduce_mean(xt, axis=0)
	xs=tf.expand_dims(xs,axis=-1)
	xs = tf.expand_dims(xs, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xs_1=tf.transpose(xs,[0,2,1,3])
	xs_2 = tf.transpose(xs, [0, 2, 3, 1])
	xt_1 = tf.transpose(xt, [0, 2, 1, 3])
	xt_2 = tf.transpose(xt, [0, 2, 3, 1])
	HR_Xs=xs*xs_1*xs_2   # dim: b*L*L*L
	HR_Xs=tf.reduce_mean(HR_Xs,axis=0)   #dim: L*L*L
	HR_Xt = xt * xt_1 * xt_2
	HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
	return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))

def HoMM4(xs,xt):
	ind=tf.range(tf.cast(xs.shape[1],tf.int32))
	ind=tf.random_shuffle(ind)
	xs=tf.transpose(xs,[1,0])
	xs=tf.gather(xs,ind)
	xs = tf.transpose(xs, [1, 0])
	xt = tf.transpose(xt, [1, 0])
	xt = tf.gather(xt, ind)
	xt = tf.transpose(xt, [1, 0])
	return HoMM4_loss(xs[:,:30],xt[:,:30])+HoMM4_loss(xs[:,30:60],xt[:,30:60])+HoMM4_loss(xs[:,60:90],xt[:,60:90])

def HoMM4_loss(xs, xt):
	xs = xs - tf.reduce_mean(xs, axis=0)
	xt = xt - tf.reduce_mean(xt, axis=0)
	xs = tf.expand_dims(xs,axis=-1)
	xs = tf.expand_dims(xs, axis=-1)
	xs = tf.expand_dims(xs, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xs_1 = tf.transpose(xs,[0,2,1,3,4])
	xs_2 = tf.transpose(xs, [0, 2, 3, 1,4])
	xs_3 = tf.transpose(xs, [0, 2, 3, 4, 1])
	xt_1 = tf.transpose(xt, [0, 2, 1, 3,4])
	xt_2 = tf.transpose(xt, [0, 2, 3, 1,4])
	xt_3 = tf.transpose(xt, [0, 2, 3, 4, 1])
	HR_Xs=xs*xs_1*xs_2*xs_3    # dim: b*L*L*L*L
	HR_Xs=tf.reduce_mean(HR_Xs,axis=0)  # dim: L*L*L*L
	HR_Xt = xt * xt_1 * xt_2*xt_3
	HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
	return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))