##############################
# Package Imports.           #
##############################
# Standard library modules.
import os
import datetime
import itertools

# Third party modules.
import numpy
import imageio
import skimage

# Local modules.
import inputs
from .Models.discriminator import define_discriminator
from .Models.generator     import define_generator
from .Models.gan           import define_gan
from .results              import NormalizationResults
from .preprocess           import image2train, train2image

#############################
# Internal Functions.       #
#############################
def _create_models(config):
    seed     = config.getint("NORMALIZATION", "seed")
    height   = config.getint("NORMALIZATION", "height")
    width    = config.getint("NORMALIZATION", "width")
    channels = config.getint("NORMALIZATION", "channels")
    shape    = (height, width, channels)

    d_model   = define_discriminator(shape, seed)
    g_model   = define_generator(shape, seed)
    gan_model = define_gan(g_model, d_model, shape)

    return d_model, g_model, gan_model

def _create_batch(config):
    train_combined_path = config.get("NORMALIZATION",    "train_combined_path")
    output_type         = config.get("NORMALIZATION",    "output_type")
    batch_size          = config.getint("NORMALIZATION", "batch_size")
    channels            = config.getint("NORMALIZATION", "channels")

    # Choose <batch_size> images from the batch.
    paths = inputs.gen_dataset(train_combined_path, output_type)
    for path, image in itertools.islice(paths, batch_size):
        image   = image.astype(float)
        h, w, _ = image.shape
        half_w  = int(w / 2)

        # Convert to color and grayscale
        color     = image[:, :half_w, :channels]
        grayscale = image[:, half_w:, :channels]

        # Manipulate images to fit models.
        color     = skimage.transform.resize(color,     (h, half_w), mode='reflect', anti_aliasing=True)
        grayscale = skimage.transform.resize(grayscale, (h, half_w), mode='reflect', anti_aliasing=True)
        if numpy.random.random() > 0.5:
            color     = numpy.fliplr(color)
            grayscale = numpy.fliplr(grayscale)

        yield image2train(color), image2train(grayscale)

def _train_epoch(window, config, epoch):
    batch_size  = config.getint("NORMALIZATION", "batch_size")
    models_path = config.get("NORMALIZATION",    "models_path")

    results = NormalizationResults(epoch)
    d_model, g_model, gan_model = _create_models(config)

    # Determine the output square shape of the discriminator.
    n_patch = d_model.output_shape[1]

    # Calculate the number of training iterations.
    y_real = numpy.ones((1, n_patch, n_patch, 1))

    # Select a batch of real samples.
    window.norm_progress.set(0)
    results.start_time = datetime.datetime.now()
    for index, (X_realB, X_realA) in enumerate(_create_batch(config)):

        # Generate a batch of fake samples.
        X_fakeB = g_model.predict(X_realA)
        y_fake  = numpy.zeros((len(X_fakeB), n_patch, n_patch, 1))

        # Update models for real samples.
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # Update the generator.
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # Save the loss values in the array.
        results.disc_real_loss.append(d_loss1)
        results.disc_fake_loss.append(d_loss2)
        results.generator_loss.append(g_loss)
        window.norm_progress.set(100 * (index + 1) / batch_size)

        # Save the models over the minimum loss generator.
        if g_loss <= min(results.generator_loss):
            d_model.save(os.path.join(models_path,   f"d_model_{epoch}.keras"))
            g_model.save(os.path.join(models_path,   f"g_model_{epoch}.keras"))
            gan_model.save(os.path.join(models_path, f"gan_model_{epoch}.keras"))

    results.end_time       = datetime.datetime.now()
    results.elapsed_time   = results.end_time - results.start_time
    results.disc_real_loss = numpy.array(results.disc_real_loss)
    results.disc_fake_loss = numpy.array(results.disc_fake_loss)
    results.generator_loss = numpy.array(results.generator_loss)

    return results, (d_model, g_model, gan_model)

def _save_results(config, epoch, results):
    results_path = config.get("NORMALIZATION", "results_path")
    numpy.save(os.path.join(results_path, f'E{epoch}_Discriminator_real_loss.npy'), results.disc_real_loss)
    numpy.save(os.path.join(results_path, f'E{epoch}_Discriminator_fake_loss.npy'), results.disc_fake_loss)
    numpy.save(os.path.join(results_path, f'E{epoch}_Generator_loss.npy'),          results.generator_loss)
    with open(os.path.join(results_path, f'E{epoch}_results.txt'), 'w+') as filehandle:
        filehandle.write(results.start_time.strftime("Start Time: %d-%b-%Y (%H:%M:%S.%f)\n"))
        filehandle.write(results.end_time.strftime("End Time: %d-%b-%Y (%H:%M:%S.%f)\n"))
        filehandle.write(f"Elapsed Time: {str(results.elapsed_time)}")

#############################
# API Functions.            #
#############################
def train(window, config):
    batch_size = config.getint("NORMALIZATION", "batch_size")
    train_size = config.getint("NORMALIZATION", "train_size")
    epochs     = config.getint("NORMALIZATION", "epochs")
    batch_size = min(batch_size, train_size)
    config.set("NORMALIZATION", "batch_size", str(batch_size))

    if config.has_option("NORMALIZATION", "trained"):
        return None
    config.set("NORMALIZATION", "trained", "True")

    for epoch in range(1, epochs + 1):
        window.norm_status.set(f"Training epoch {epoch}/{epochs} ({batch_size})")
        results, models = _train_epoch(window, config, epoch)
        _save_results(config, epoch, results)