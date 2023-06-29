import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
import geode.datasets as gd
import geode.models as gm
from tensorflow.keras import backend as K
from geode.losses import dice_loss, iou_loss
from tensorflow.keras.losses import CategoricalCrossentropy

# these lines fix memory errors
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# this line allows for mixed-precision computations
tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

# these lines suppress an annoying warning when using tf.data.Datasets
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

# tensorflow always 0-indexes whichever GPUs are visible
strategy = tf.distribute.MirroredStrategy(devices=['GPU:' + str(n) for n in [0, 1, 2, 3]])

# set the model parameters
N_CHANNELS = 3
N_FILTERS = 64
DROPOUT_RATE = 0.3

# set the training parameters
N_EPOCHS = 500
BATCH_SIZE = 60
LEARNING_RATE = 0.001

# set up the data generators
TILE_DIMENSION = 512
TRAIN_TILES_PATH = '/home/rdgrlmpr/training_datasets/Inria/Train'
#TRAIN_STEPS_PER_EPOCH = 50
TRAIN_STEPS_PER_EPOCH = int(len(os.listdir(os.path.join(TRAIN_TILES_PATH, 'imagery'))) / BATCH_SIZE)
VAL_TILES_PATH = '/home/rdgrlmpr/training_datasets/Inria/Validation'
#VAL_STEPS_PER_EPOCH = 50
VAL_STEPS_PER_EPOCH = int(len(os.listdir(os.path.join(VAL_TILES_PATH, 'imagery'))) / BATCH_SIZE)
PERFORM_ONE_HOT = True

# set up the test dataset info
TEST_IMAGERY_PATH = '/home/rdgrlmpr/test_datasets/conus_wv3/rgb'
TEST_LABELS_PATH = '/home/rdgrlmpr/test_datasets/conus_wv3/labels'
TEST_PREDICTIONS_PATH = 'predictions'

# set up the callbacks
CSV_LOGGER_PATH = 'training_log.csv'
ES_MONITOR = 'val_loss'
ES_MIN_DELTA = 0.0001
ES_PATIENCE = 100
LR_MONITOR = 'val_loss'
LR_FACTOR = 0.5
LR_PATIENCE = 20
CHECKPOINT_PATH = 'saved_model'
CHECKPOINT_MONITOR = 'val_loss'

# initialize the data generators
train_ds = gd.SegmentationDataset(tile_dimension=TILE_DIMENSION,
                                  tiles_path=TRAIN_TILES_PATH,
                                  n_channels=N_CHANNELS)

train_gen = train_ds.tf_dataset(augmentation=True,
						  batch_size=BATCH_SIZE,
                                perform_one_hot=PERFORM_ONE_HOT).with_options(options)

val_ds = gd.SegmentationDataset(tile_dimension=TILE_DIMENSION,
                                tiles_path=VAL_TILES_PATH,
                                n_channels=N_CHANNELS)

val_gen = val_ds.tf_dataset(augmentation=False,
                            batch_size=BATCH_SIZE,
                            perform_one_hot=PERFORM_ONE_HOT).with_options(options)

# initialize the callbacks
my_callbacks = [CSVLogger(filename=CSV_LOGGER_PATH, append=True),
                EarlyStopping(monitor=ES_MONITOR, min_delta=ES_MIN_DELTA, patience=ES_PATIENCE),
                ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor=CHECKPOINT_MONITOR, save_best_only=True),
                ReduceLROnPlateau(monitor=LR_MONITOR, factor=LR_FACTOR, patience=LR_PATIENCE),
                TerminateOnNaN()]

# set up the GPUs for training
with strategy.scope():
	LOSS_FUNCTION = iou_loss
	cnn = gm.VGG19Unet(n_filters=N_FILTERS, n_classes=2, dropout_rate=DROPOUT_RATE)
	cnn.compile_model(loss=LOSS_FUNCTION, learning_rate=LEARNING_RATE, include_residual=True)
	cnn.model.compile(optimizer='adam', loss=LOSS_FUNCTION)

# train the model
cnn.model.fit(x=train_gen,
              epochs=N_EPOCHS,
              steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
              validation_data=val_gen,
              validation_steps=VAL_STEPS_PER_EPOCH,
              callbacks=my_callbacks)

# predict test imagery and compute metrics
cnn.model = tf.keras.models.load_model('saved_model', custom_objects={'iou_loss': iou_loss})
cnn.predict_test_imagery(test_imagery_path=TEST_IMAGERY_PATH,
                         test_labels_path=TEST_LABELS_PATH,
                         test_predictions_path=TEST_PREDICTIONS_PATH)
