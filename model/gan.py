import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dropout, Dense, MaxPooling2D, \
    Reshape, LeakyReLU, UpSampling2D, BatchNormalization, ReLU
from keras.optimizers import Adam


class GAN:
    def __init__(self, input_dims=(128,128,1), latent_dim=100, embedding_dims=(8,8), save_dir='saved'):

        self.input_dims = input_dims
        self.n_channels = input_dims[-1]

        self.latent_dim = latent_dim
        self.embedding_dims = embedding_dims
        self.save_dir = Path(save_dir)

        self.discriminator = None
        self.generator = None
        self.gan = None
        self.dataset = None

        self.save_dir.mkdir(exist_ok=True)

    def define_discriminator(self):

        model = Sequential()

        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same', input_shape=self.input_dims))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(.35))

        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(.35))

        model.add(Flatten())
        model.add(Dropout(.4))

        model.add(Dense(1, activation='sigmoid'))

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.summary()

        self.discriminator = model
        return model

    def define_generator(self):

        n_nodes = 128 * np.prod(self.embedding_dims)

        model = Sequential()
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Reshape((*self.embedding_dims, 128)))

        model.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(.35))

        model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(.4))

        model.add(Conv2D(self.n_channels, (3,3), activation='tanh', padding='same'))

        model.summary()

        self.generator = model
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self):
        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self.generator)
        # add the discriminator
        model.add(self.discriminator)
        # compile model

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        model.summary()

        self.gan = model
        return model


    def load_dataset(self, path):
        self.dataset = np.load(path)['images']
        self.class_labels = np.load(path)['labels']

        return self.dataset, self.class_labels

    def generate_real_samples(self, n_samples):
        ix = np.random.randint(0, self.dataset.shape[0], n_samples)

        return self.dataset[ix], np.ones((n_samples, 1))

    def generate_latent_points(self, n):
        # generate points in the latent space
        x_input = np.random.randn(self.latent_dim * n)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, self.latent_dim)

        return x_input

    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = self.generator.predict(x_input)

        # create class labels
        y = np.zeros((n_samples, 1))
        return X, y

    def train(self, n_epochs=100, n_batch=128):

        bat_per_epo = int(self.dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)

        run_stats = {
            'epoch': [],
            'batch': [],
            'd1': [],
            'd2': [],
            'g': []
        }

        target_dir = self.save_dir / f'{datetime.now().strftime("%y-%m-%d_%H-%M-%S")}'
        target_dir.mkdir()

        for i in tqdm(range(n_epochs)):
            # enumerate batches over the training set

            for j in range(bat_per_epo):

                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # update discriminator model weights
                d_loss1, _ = self.discriminator.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(half_batch)
                # update discriminator model weights
                d_loss2, _ = self.discriminator.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch(X_gan, y_gan)

                # save run stats
                run_stats['epoch'].append(i)
                run_stats['batch'].append(j)
                run_stats['d1'].append(d_loss1)
                run_stats['d2'].append(d_loss2)
                run_stats['g'].append(g_loss)

                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

            # produce a batch of fakes to track progress
            self.produce_fakes(10)

            if i % 10 == 0:
                # save the generator model
                self.generator.save(target_dir / 'generator.h5')

                # save the run statistics
                run_df = pd.DataFrame.from_dict(run_stats)
                run_df.to_csv(target_dir / 'stats.csv')

                # save a plot of the run statistics
                run_df['x'] = (run_df['epoch']) * bat_per_epo + run_df['batch']
                sns.lineplot(x='x', y='value', hue='variable',
                             data=pd.melt(run_df[['x', 'd1', 'd2', 'g']], 'x'))

                plt.ylim(0,10)

                plt.savefig(target_dir / 'd_stats.jpg', dpi=300)
                plt.show()

    def load_generator(self, path):
        self.generator = load_model(path)

        print('loaded generator')

    def produce_fakes(self, n, save_dir='../model/output/'):

        Path(save_dir).mkdir(exist_ok=True)

        fakes, _ = self.generate_fake_samples(n)

        for i, img in enumerate(fakes):

            img = (img + 1) * 127.5
            if self.n_channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img.astype('int')


            cv2.imwrite(save_dir + f'{i}.jpg', img)


if __name__ == '__main__':

    # can try some methods from these implementations
    # https://developers.google.com/machine-learning/gan ( new loss function, minimax or wasserschutz loss )
    # https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
    # https://jonathan-hui.medium.com/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f (some tips)
    # https://github.com/MrForExample/Generative_Models_Collection

    # Other ideas:
        # longer training (100it)
        # eliminate fully ocnnected (see DCGAN)
        # testing with batch norm? can explore spectral norm etc.
        # explore implementation with hinge loss? (see 4)
        # can explore using_gpu_memory_growth (see 4)

    test_model = GAN()

    test_model.define_generator()
    test_model.define_discriminator()
    test_model.define_gan()

    test_model.load_dataset('../dataset/processed/24335img_BW.npz')

    print('training...')
    test_model.train()

    # generate fake landscapes
    test_model.produce_fakes(100)
