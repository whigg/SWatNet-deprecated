import tensorflow as tf
import config
import time
import data_loader
from models.unet_gru_triple import unet_gru_triple
from models.unet_triple import unet_triple
from models.unet_triple import unet_triple_v2

@tf.function
def train_step(model, loss_fun, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_pre,_,_,_ = model(x, training=True)
        loss = loss_fun(y, y_pre)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    config.tra_loss_tracker.update_state(loss)
    config.tra_oa.update_state(y, y_pre)
    config.tra_miou.update_state(y, y_pre)
    return config.tra_loss_tracker.result(), config.tra_oa.result(), config.tra_miou.result()

@tf.function
def test_step(model, loss_fun, x, y):
    with tf.GradientTape() as tape:
        y_pre,_,_,_ = model(x, training=False)
        loss = loss_fun(y, y_pre)
    config.test_loss_tracker.update_state(loss)
    config.test_oa.update_state(y, y_pre)
    config.test_miou.update_state(y, y_pre)
    return config.test_loss_tracker.result(), config.test_oa.result(), config.test_miou.result()

def train_loops(model, loss_fun, optimizer, tra_dset, test_dset, epochs):
    max_miou_pre = 0.7
    for epoch in range(epochs):
        start = time.time()
        # train the model
        for x_batch, y_batch in tra_dset:            
            tra_loss_epoch,tra_oa_epoch,tra_miou_epoch = train_step(model, loss_fun, optimizer, x_batch, y_batch)
        # test the model
        for x_batch, y_batch in test_dset:
            test_loss_epoch, test_oa_epoch, test_miou_epoch = test_step(model, loss_fun, x_batch, y_batch)
        config.tra_loss_tracker.reset_states(), config.tra_oa.reset_states(), config.tra_miou.reset_states()
        config.test_loss_tracker.reset_states(), config.test_oa.reset_states(), config.test_miou.reset_states()
        ### write into tensorboard
        ## tensorboard writer
        # train_summary_writer = tf.summary.create_file_writer(config.train_log_dir)
        # test_summary_writer = tf.summary.create_file_writer(config.test_log_dir)
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('learning rate', data=config.optimizer.learning_rate(epoch*16), step=epoch)
        #     tf.summary.scalar('loss', data=tra_loss_epoch, step=epoch)
        #     tf.summary.scalar('oa', data=tra_oa_epoch, step=epoch)
        #     tf.summary.scalar('miou', data=tra_miou_epoch, step=epoch)
        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss', data=test_loss_epoch, step=epoch)
        #     tf.summary.scalar('oa', data=test_oa_epoch, step=epoch)
        #     tf.summary.scalar('miou', data=test_miou_epoch, step=epoch)
        # print the metrics
        print('epoch {}: traLoss:{:.3f}, traOA:{:.2f}, traMIoU:{:.2f}; evaLoss:{:.3f}, evaOA:{:.2f}, evaMIoU:{:.2f}, time:{:.0f}s'.format(epoch + 1, tra_loss_epoch, tra_oa_epoch, tra_miou_epoch, test_loss_epoch, test_oa_epoch, test_miou_epoch, time.time() - start))
        ## save the temporal trained weights
        if test_miou_epoch>max_miou_pre:
            max_miou_pre = test_miou_epoch
            model.save_weights(config.path_savedmodel+'/unet_gru_triple/weights_epoch_%d'%(epoch+1))

if __name__ == '__main__':
    '''load dataset'''
    tra_dset = data_loader.get_tra_dset()
    test_dset = data_loader.get_eva_dset()
    '''training configuration'''
    loss_fun = config.binary_ce_loss
    optimizer = config.optimizer
    model = unet_gru_triple(scale_high=2048, scale_mid=512, scale_low=256, nclass=2,
                                        trainable_gru=True, trainable_unet=True)
    # model = unet_triple(scale_high=2048, scale_mid=512, scale_low=256, nclass=2)
    # model = unet_triple_v2(scale_high=2048, scale_mid=512, scale_low=256, nclass=2)
    '''model training'''
    train_loops(model, loss_fun, optimizer, tra_dset, \
        test_dset, epochs=config.epoches)
    '''model save'''
    path_save_model = root_dir + '/models/pretrained/unet_gru_triple'
    model.save(path_save_model)
