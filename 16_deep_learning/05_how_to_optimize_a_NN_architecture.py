#!/usr/bin/env python
# coding: utf-8

# # Train a Deep NN to predict Asset Price movements

# In practice, we need to explore variations of the design options outlined above because we can rarely be sure from the outset which network architecture best suits the data.
# 
# The GridSearchCV class provided by scikit-learn that we encountered in Chapter 6, The Machine Learning Workflow conveniently automates this process. Just be mindful of the risk of false discoveries and keep track of how many experiments you are running to adjust the results accordingly.
# 
# In this section, we will explore various options to build a simple feedforward Neural Network to predict asset price moves for a one-month horizon.

# ## Setup Docker for GPU acceleration

# `docker run -it -p 8889:8888 -v /path/to/machine-learning-for-trading/16_convolutions_neural_nets/cnn:/cnn --name tensorflow tensorflow/tensorflow:latest-gpu-py3 bash`

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


import os
from pathlib import Path
from importlib import reload
from joblib import dump, load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint


# In[2]:


np.random.seed(42)


# ## Create a stock return series to predict asset price moves

# We will use the last 24 monthly returns and dummy variables for the month and the year to predict whether the price will go up or down the following month. We use the daily Quandl stock price dataset (see GitHub for instructions on how to source the data).

# In[3]:


prices = (pd.read_hdf('../data/assets.h5', 'quandl/wiki/prices')
          .adj_close
          .unstack().loc['2007':])
prices.info()


# We will work with monthly returns to keep the size of the dataset manageable and remove some of the noise contained in daily returns, which leaves us with almost 2,500 stocks with 120 monthly returns each:

# In[4]:


returns = (prices
           .resample('M')
           .last()
           .pct_change()
           .loc['2008': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns.info()


# In[5]:


returns.head().append(returns.tail())


# In[6]:


n = len(returns)
T = 24
tcols = list(range(25))


# In[7]:


data = pd.DataFrame()
for i in range(n-T-1):
    df = returns.iloc[i:i+T+1]
    data = pd.concat([data, (df.reset_index(drop=True).T
                             .assign(year=df.index[0].year,
                                     month=df.index[0].month))],
                     ignore_index=True)
data[tcols] = (data[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))
data['label'] = (data[0] > 0).astype(int)
data['date'] = pd.to_datetime(data.assign(day=1)[['year', 'month', 'day']])
data = pd.get_dummies((data.drop(0, axis=1)
                       .set_index('date')
                       .apply(pd.to_numeric)), 
                      columns=['year', 'month']).sort_index()
data.info()


# In[8]:


data.to_hdf('data.h5', 'returns')


# In[9]:


data.shape


# In[12]:


class OneStepTimeSeriesSplit:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled 'date'"""

    def __init__(self, n_splits=3, test_period_length=1, shuffle=False):
        self.n_splits = n_splits
        self.test_period_length = test_period_length
        self.shuffle = shuffle
        self.test_end = n_splits * test_period_length

    @staticmethod
    def chunks(l, chunk_size):
        for i in range(0, len(l), chunk_size):
            yield l[i:i + chunk_size]

    def split(self, X, y=None, groups=None):
        unique_dates = (X.index
                            .get_level_values('date')
                            .unique()
                            .sort_values(ascending=False)[:self.test_end])

        dates = X.reset_index()[['date']]
        for test_date in self.chunks(unique_dates, self.test_period_length):
            train_idx = dates[dates.date < min(test_date)].index
            test_idx = dates[dates.date.isin(test_date)].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# ## Define Network Architecture

# ### Custom AUC Loss Metric

# For binary classification, AUC is an excellent metric because it assesses performance irrespective of the threshold chosen to convert probabilities into positive predictions. Unfortunately, Keras does not provide it ‘out-of-the-box’ because it focuses on metrics that help gradient descent optimized based on batches of samples during training. However, we can define a custom loss metric for use with the early stopping callback as follows (included in the compile step):

# In[20]:


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


# ### Set up `build_fn` for `keras.wrappers.scikit_learn.KerasClassifier`

# Keras contains a wrapper that we can use with the sklearn GridSearchCV class. It requires a build_fn that constructs and compiles the model based on arguments that can later be passed during the GridSearchCV iterations.
# 
# The following `make_model` function illustrates how to flexibly define various architectural elements for the search process. The dense_layers argument defines both the depth and width of the network as a list of integers. We also use dropout for regularization, expressed as a float in the range [0, 1] to define the probability that a given unit will be excluded from a training iteration.

# In[78]:


def make_model(dense_layers, activation, dropout):
    '''Creates a multi-layer perceptron model
    
    dense_layers: List of layer sizes; one number per layer
    '''

    model = Sequential()
    for i, layer_size in enumerate(dense_layers, 1):
        if i == 1:
            model.add(Dense(layer_size, input_dim=input_dim))
            model.add(Activation(activation))
        else:
            model.add(Dense(layer_size))
            model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['binary_accuracy', auc_roc])

    return model


# ## Run Keras with `GridSearchCV`

# ### Train-Test Split

# We split the data into a training set for cross-validation, and keep the last 12 months with data as holdout test:

# In[9]:


data = pd.read_hdf('data.h5', 'returns')


# In[6]:


X_train = data[:'2016'].drop('label', axis=1)
y_train = data[:'2016'].label


# In[7]:


X_test = data['2017':].drop('label', axis=1)
y_test = data['2017':].label


# ### Define GridSearch inputs

# Now we just need to define our Keras classifier using the make_model function, set cross-validation (see chapter 6 on The Machine Learning Process and following for the OneStepTimeSeriesSplit), and the parameters that we would like to explore. 
# 
# We pick several one- and two-layer configurations, relu and tanh activation functions, and different dropout rates. We could also try out different optimizers (but did not run this experiment to limit what is already a computationally intensive effort):

# In[ ]:


input_dim = X_train.shape[1]


# In[62]:


clf = KerasClassifier(make_model, epochs=10, batch_size=32)


# In[13]:


n_splits = 12


# In[14]:


cv = OneStepTimeSeriesSplit(n_splits=n_splits)


# In[60]:


param_grid = {'dense_layers': [[32], [32, 32], [64], [64, 64], [64, 64, 32], [64, 32], [128]],
              'activation'  : ['relu', 'tanh'],
              'dropout'     : [.25, .5, .75],
              }


# To trigger the parameter search, we instantiate a GridSearchCV object, define the fit_params that will be passed to the Keras model’s fit method, and provide the training data to the GridSearchCV fit method:

# In[64]:


gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='roc_auc',
                  cv=cv,
                  refit=True,
                  return_train_score=True,
                  n_jobs=-1,
                  verbose=1,
                  iid=False,
                  error_score=np.nan)


# In[ ]:


fit_params = dict(callbacks=[EarlyStopping(monitor='auc_roc', 
                                           patience=300, 
                                           verbose=1, mode='max')],
                  verbose=2,
                  epochs=50)


# In[ ]:


gs.fit(X=X_train.astype(float), y=y_train, **fit_params)
print('\nBest Score: {:.2%}'.format(gs.best_score_))
print('Best Params:\n', pd.Series(gs.best_params_))


# ### Persist best model and training data

# In[ ]:


gs.best_estimator_.model.save('best_model.h5')


# In[ ]:


pd.DataFrame(gs.cv_results_).to_csv('cv_results.csv', index=False)


# In[ ]:


y_pred = gs.best_estimator_.model.predict(test_data.drop('label', axis=1))
roc_auc_score(y_true=test_data.label, y_score=y_pred)


# In[9]:


with pd.HDFStore('data.h5') as store:
    store.put('X_train', X_train)
    store.put('X_test', X_test)
    store.put('y_train', y_train)
    store.put('y_test', y_test)


# In[94]:


cv_results = pd.read_csv('gridsearch/cv_results.csv')
cv_results = (cv_results.filter(like='param_')
              .join(cv_results
                    .filter(like='_test_score')
                    .filter(like='split'))
             .rename(columns = lambda x: x.replace('param_', '')))
cv_results =pd.melt(id_vars=['activation', 'dense_layers', 'dropout'], 
                    frame=cv_results,
                   value_name='score',
                   var_name='split')
cv_results.info()


# The following chart shows the range of cross-validation results for the various elements of the Neural Network architectures that we tested in our experiment. It shows that the settings that performed best in combination, when evaluated individually, tended to do as good as or better than the alternatives.

# In[119]:


fig = plt.figure(constrained_layout=True, figsize=(14, 6))
gs = GridSpec(nrows=1, ncols=4, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel('Activation Functon')
sns.boxenplot(x='activation', y='score', data=cv_results, ax=ax1)
ax2 = fig.add_subplot(gs[0, 1])
sns.boxenplot(x='dropout', y='score', data=cv_results, ax=ax2);
ax2.set_xlabel('Dropout Rate')
ax3 = fig.add_subplot(gs[0, 2:])
sns.boxenplot(x='dense_layers', y='score', data=cv_results, ax=ax3)
ax3.set_xlabel('Hidden Layers')
fig.suptitle('Performance Impact of Architecture Elements', fontsize=16)
fig.savefig('parameter_impact', dpi=300);


# ## Load best model

# In[8]:


model = load_model('gridsearch/best_model.h5', custom_objects={'auc_roc': auc_roc})


# In[9]:


model.summary()


# ### Predict 1 year of price moves

# In[ ]:


y_pred = model.predict(test_data.drop('label', axis=1))


# In[11]:


roc_auc_score(y_score=y_pred, y_true=test_data.label)


# ## Retrain with all data

# ### Custom ROC AUC Callback

# In[7]:


class auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(y_true=self.y, y_score=y_pred)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(y_true=self.y_val, y_score=y_pred_val)
        print('\rroc-auc: {:.2%} - roc-auc_val: {:.2%}'.format(roc, roc_val),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# ### Early Stopping

# In[18]:


early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=5,
                               verbose=0,
                               mode='auto',
                               baseline=None,
                               restore_best_weights=False)


# ### Model Checkpoints

# In[19]:


checkpointer = ModelCheckpoint('models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                               monitor='val_loss',
                               verbose=0,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               period=1)


# ### Tensorboard

# In[20]:


tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=1,
                          batch_size=32,
                          write_graph=True,
                          write_grads=True,
                          update_freq='epoch')


# In[10]:


data = pd.read_hdf('data.h5', 'returns')
features = data.drop('label', axis=1)
label = data.label


# ### Run cross-validation

# In[31]:


for fold, (train_idx, test_idx) in enumerate(cv.split(data)):
    checkpointer = ModelCheckpoint('models/weights.{}.hdf5'.format(fold),
                               monitor='val_loss',
                               verbose=0,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               period=1)
    tensorboard = TensorBoard(log_dir='./logs/{}'.format(fold),
                          histogram_freq=1,
                          batch_size=32,
                          write_graph=True,
                          write_grads=True,
                          update_freq='epoch')
    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]
    y_train = label.iloc[train_idx]
    y_test = label.iloc[test_idx]

    training = model.fit(X_train, 
                         y_train, 
                          batch_size=32, 
                          epochs=50, 
                          verbose=1, 
                          validation_data=(X_test, y_test), 
                          callbacks=[checkpointer, 
                                     tensorboard,
                                     early_stopping,
                                     auc_callback(training_data=(X_train, y_train),
                                                  validation_data=(X_test, y_test))])
    history = pd.concat([history, pd.DataFrame(training.history).assign(fold=fold)])


# In[29]:


scores, preds = {}, {}
for fold, (train_idx, test_idx) in enumerate(cv.split(data)):
    model = load_model(f'models/weights.{fold}.hdf5', custom_objects={'auc_roc': auc_roc})
    y_test = features.iloc[test_idx]
    month = y_test.index[0].month
    preds[month] = model.predict(y_test)
    scores[month] = roc_auc_score(y_score=preds[month], y_true=label.iloc[test_idx])


# In[27]:


pd.Series(scores).sort_index().plot.bar();


# ### Make Predictions

# In[38]:


predictions = pd.DataFrame({month: data.squeeze() for month, data in preds.items()}, index = range(preds[1].shape[0])).sort_index(1)
predictions.info()


# ### Evaluate Results

# In[123]:


from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score


# In[125]:


bins = np.arange(0, 1.01, .01)
roc, prc = pd.Series(), pd.Series()
avg_roc, avg_precision = [], []
for month, y_score in predictions.items():
    y_true = label[f'2017{month:02}01']
    avg_roc.append(roc_auc_score(y_true=y_true, y_score=y_score))
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df.fpr = pd.cut(df.fpr, bins=bins, labels=bins[1:])
    roc = pd.concat([roc, df.groupby('fpr').tpr.mean().bfill().to_frame('tpr').reset_index()])
    
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    avg_precision.append(average_precision_score(y_true=y_true, y_score=y_score))
    df = pd.DataFrame({'precision': precision, 'recall': recall})
    df.recall = pd.cut(df.recall, bins=bins, labels=bins[1:])
    prc = pd.concat([prc, df.groupby('recall').precision.mean().ffill().to_frame('precision').reset_index()])
    


# In[126]:


np.mean(avg_roc), np.mean(avg_precision)


# To obtain a measure of the model’s generalization error, we evaluate its predictive performance on the hold-out set. To this end, we iteratively predict one month in the test after training the best-performing architecture on all preceding months.
# 
# The below ROC and Precision-Recall curves summarize the out-of-sample performance over the 12 months in 2017. The average AUC score is 0.7739, and the average precision is 68.8%, with the full range of the tradeoffs represented by the two graphs.

# While the AUC scores underline solid predictive performance, we need to be careful because binary price moves ignore the size of the moves. We would need to deepen our analysis to understand whether good directional predictions would translate into a profitable trading strategy.

# In[129]:


fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
sns.lineplot(x='fpr', y='tpr', data=roc, ax=axes[0])
pd.Series(bins, index=bins).plot(ax=axes[0], ls='--', lw=1, c='k')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].text(x=.05, y=.94, s=f'Average AUC: {np.mean(avg_roc):.2%}')
sns.lineplot(x='recall', y='precision', data=prc, ax=axes[1])
axes[1].set_title('Precision-Recall Curve')
axes[1].text(x=.65, y=.9, s=f'Average Precision: {np.mean(avg_precision):.2%}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].axhline(.5, ls='--', lw=1, c='k')
fig.suptitle('2-Layer Feedforward Network: Stock Price Movement Prediction', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=.86)
fig.savefig('figures/roc_prc_curves', dpi=300);


# ### How to further improve the results
# 
# The relatively simple architecture yields some promising results. To further improve performance, you can
# - First and foremost, add new features and more data to the model
# - Expand the set of architectures to explore, including more or wider layers
# - Inspect the training progress and train for more epochs if the validation error continued to improve at 50 epochs
# 
# Finally, you can use more sophisticated architectures, including Recurrent Neural Networks (RNN) and Convolutional Neural Networks that are well suited to sequential data, whereas vanilla feedforward NNs are not designed to capture the ordered nature of the features.
# 
