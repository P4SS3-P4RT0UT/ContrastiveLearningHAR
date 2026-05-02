# %% [markdown]
# # SimCLR Contrastive Training for Human Activity Recognition Tutorial

# %%
# Author: C. I. Tang
# Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
# Contact: cit27@cl.cam.ac.uk
# License: GNU General Public License v3.0

# %% [markdown]
# ## Imports

# %%
import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# %%
# Library scripts
import raw_data_processing
import data_pre_processing
import simclr_models
import simclr_utitlities
import transformations

# %%
working_directory = 'uci_har_run/'
dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

# %% [markdown]
# ## UCI-HAR Dataset
# 
# In this section, the UCI-HAR dataset will be downloaded and parsed. The results will then be saved in a python pickle file.
# (Note: This section only needs to be run once)
# 
# Citation:
# ```
# @article{anguita2013public,
#   title={A public domain dataset for human activity recognition using smartphones},
#   author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
#   journal={Esann},
#   volume={3},
#   pages={3},
#   year={2013}
# }
# ```

# %% [markdown]
# ### Downloading & Unzipping

# %%
import requests
import zipfile

# %%

with zipfile.ZipFile(working_directory + 'UCI_HAR_Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall(working_directory)

# %% [markdown]
# ## Pre-processing

# %% [markdown]
# Here we split it into training, validation and testing sets. 

# %%
accelerometer_data_folder_path = working_directory + 'UCI HAR Dataset/'
user_datasets = raw_data_processing.process_uci_har_accelerometer_files(accelerometer_data_folder_path)

# %%
with open(working_directory + 'uci_har_user_split.pkl', 'wb') as f:
    pickle.dump({
        'user_split': user_datasets,
    }, f)

# %%
window_size = 128  
input_shape = (window_size, 3)

dataset_name = 'uci_har.pkl'
dataset_name_user_split = 'uci_har_user_split.pkl'

label_list = ['null', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
label_list_full_name = ['null', 'walking', 'walking upstairs', 'walking downstairs', 'sitting', 'standing', 'laying']
has_null_class = True

label_map = dict([(i, i) for i in range(len(label_list))])

output_shape = len(label_list)

model_save_name = f"uci_har_acc"

sampling_rate = 50.0 

# a fixed user-split

test_users_fixed = [2, 4, 9, 10, 12, 13, 18, 20, 24] # predefined test users for UCI-HAR
def get_fixed_split_users(har_users):
    test_users = test_users_fixed
    train_users = [u for u in har_users if u not in test_users]
    return (train_users, test_users)

# %%
with open(dataset_save_path + dataset_name_user_split, 'rb') as f:
    dataset_dict = pickle.load(f)
    user_datasets = dataset_dict['user_split']

# %%
har_users = list(user_datasets.keys())
train_users, test_users = get_fixed_split_users(har_users)

# %%
np_train, np_val, np_test = data_pre_processing.pre_process_dataset_composite(
    user_datasets=user_datasets, 
    label_map=label_map, 
    output_shape=output_shape, 
    train_users=train_users, 
    test_users=test_users, 
    window_size=window_size, 
    shift=window_size//2, 
    normalise_dataset=True, 
    verbose=1
)

# %%
print(np_train[0].shape)

# %% [markdown]
# ## SimCLR Training

# %%
batch_size = 512
decay_steps = 1000
epochs = 200
temperature = 0.1
transform_funcs = [
    transformations.noise_transform_vectorized,
    # transformations.scaling_transform_vectorized, # Use Scaling trasnformation
    # transformations.rotation_transform_vectorized # Use rotation trasnformation
]
transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)

# trasnformation_indices = [2] # Use rotation trasnformation only
# trasnformation_indices = [1, 2] # Use Scaling and rotation trasnformation

# trasnform_funcs_vectorized = [
#     transformations.noise_transform_vectorized, 
#     transformations.scaling_transform_vectorized, 
#     transformations.rotation_transform_vectorized, 
#     transformations.negate_transform_vectorized, 
#     transformations.time_flip_transform_vectorized, 
#     transformations.time_segment_permutation_transform_improved, 
#     transformations.time_warp_transform_low_cost, 
#     transformations.channel_shuffle_transform_vectorized
# ]
# transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']



# %%
start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = simclr_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()

trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

simclr_model_save_path = f"{working_directory}{start_time_str}_simclr.keras"
trained_simclr_model.save(simclr_model_save_path)

# %%
plt.figure(figsize=(12,8))
plt.plot(epoch_losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig('epoch_losses.png')

# %% [markdown]
# ## Fine-tuning and Evaluation

# %% [markdown]
# ### Linear Model

# %%

total_epochs = 50
batch_size = 200
tag = "linear_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
linear_evaluation_model = simclr_models.create_linear_model_from_base_model(simclr_model, output_shape, intermediate_layer=7)

linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.keras"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

training_history = linear_evaluation_model.fit(
    x = np_train[0],
    y = np_train[1],
    batch_size=batch_size,
    shuffle=True,
    epochs=total_epochs,
    callbacks=[best_model_callback],
    validation_data=np_val
)

linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)

print("Model with lowest validation Loss:", flush=True)
print(simclr_utitlities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True), flush=True)
print("Model in last epoch", flush=True)
print(simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))


# %% [markdown]
# ### Full HAR Model

# %%

total_epochs = 50
batch_size = 200
tag = "full_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(simclr_model, output_shape, model_name="TPN", intermediate_layer=7, last_freeze_layer=4)

full_eval_best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.keras"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

training_history = full_evaluation_model.fit(
    x = np_train[0],
    y = np_train[1],
    batch_size=batch_size,
    shuffle=True,
    epochs=total_epochs,
    callbacks=[best_model_callback],
    validation_data=np_val
)

full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name)

print("Model with lowest validation Loss:", flush=True)
print(simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True), flush=True)
print("Model in last epoch", flush=True)
print(simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True), flush=True)

# %% [markdown]
# ## Extra: t-SNE Plots

# %% [markdown]
# ### Parameters

# %%
# Select a model from which the intermediate representations are extracted
target_model = simclr_model 
perplexity = 30.0


# %% [markdown]
# ### t-SNE Representations

# %%
intermediate_model = simclr_models.extract_intermediate_model_from_base_model(target_model, intermediate_layer=7)
intermediate_model.summary()

embeddings = intermediate_model.predict(np_test[0], batch_size=600)
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
tsne_projections = tsne_model.fit_transform(embeddings)


# %% [markdown]
# ### Plotting

# %%
labels_argmax = np.argmax(np_test[1], axis=1)
unique_labels = np.unique(labels_argmax)

plt.figure(figsize=(16,8))
graph = sns.scatterplot(
    x=tsne_projections[:,0], y=tsne_projections[:,1],
    hue=labels_argmax,
    palette=sns.color_palette("hsv", len(unique_labels)),
    s=50,
    alpha=1.0,
    rasterized=True
)
plt.xticks([], [])
plt.yticks([], [])


plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2)
legend = graph.legend_
for j, label in enumerate(unique_labels):
    legend.get_texts()[j].set_text(label_list_full_name[label]) 

plt.title(f"t-SNE plot of test set representations (perplexity={perplexity})", fontsize=16)
plt.savefig(f'tsne_plot_perplexity_{perplexity}.png', bbox_inches='tight')



# This is used to select colors for labels which are close to each other
# Each pair corresponds to one label class
# i.e. ['null', 'sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'jogging']
# The first number determines the color map, and the second determines its value along the color map
# So 'sitting', 'standing' will share similar colors, and 'walking', 'walking upstairs', 'walking downstairs' will share another set of similar colors
label_color_spectrum = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0)] 

# This step generates a list of colors for different categories of activities
# Here we assume 5 categories, and 5 different intesities within each category
major_colors = ['cool', 'Blues', 'Greens', 'Oranges', 'Purples']
color_map_base = dict (
    [((i, j), color) for i, major_color in enumerate(major_colors) for j, color in enumerate(reversed(sns.color_palette(major_color, 5))) ]
)
color_palette = np.array([color_map_base[color_index] for color_index in label_color_spectrum])

# This selects the appropriate number of colors to be used in the plot
labels_argmax = np.argmax(np_test[1], axis=1)
unique_labels = np.unique(labels_argmax)

plt.figure(figsize=(16,8))
graph = sns.scatterplot(
    x=tsne_projections[:,0], y=tsne_projections[:,1],
    hue=labels_argmax,
    palette=list(color_palette[unique_labels]),
    s=50,
    alpha=1.0,
    rasterized=True
)
plt.xticks([], [])
plt.yticks([], [])


plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2)
legend = graph.legend_
for j, label in enumerate(unique_labels):
    legend.get_texts()[j].set_text(label_list_full_name[label]) 
plt.title(f"t-SNE plot of test set representations (perplexity={perplexity})", fontsize=16)
plt.savefig(f'tsne_plot_custom_colors_perplexity_{perplexity}.png', bbox_inches='tight')


# %%



