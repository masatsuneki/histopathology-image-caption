# code adapted from https://www.tensorflow.org/tutorials/text/image_captioning
import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
import yaml


from dataclasses import dataclass, asdict
from nltk.translate.bleu_score import sentence_bleu
from wandb.integration.keras import WandbCallback


import wandb

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices("GPU")


@dataclass
class Config:
    magnification: int = 20
    path: str = f"/data/patches/x{magnification}"
    batch_size: int = 1
    extract_batch_size: int = 16
    output_path: str = f"/data"
    captions_csv_path: str = f"/data/captions.csv"
    base_model_name = "efficientnetb3"
    pooling: str = "avg"
    num_aug_repeat: int = 4
    augment: bool = False
    features_path: str = f"{output_path}/features/{base_model_name}_x{magnification}_p{pooling}{'_aug' if augment else ''}"
    embedding_dim: int = 256
    units: int = 512
    patch_size: int = 300
    wandb_mode: str = "disabled"
    wandb_project: str = "histo_image_captions"
    epochs: int = 40
    cache_features: bool = True
    learning_rate: float = 1e-3


models = {
    "efficientnetb3": {
        "model": tf.keras.applications.EfficientNetB3,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
    },
    "densenet121": {
        "model": tf.keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
}


def augment(image):
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0, upper=0.2)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(
        image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )

    return image


def extract_features_from_tiles(image_ids, train_ids, config: Config):
    feature_extractor = models[config.base_model_name]["model"](
        include_top=False, weights="imagenet"
    )

    print(feature_extractor.summary())
    if config.pooling != "avg":
        feature_extractor = tf.keras.Model(
            feature_extractor.input,
            tf.keras.layers.AveragePooling2D(int(config.pooling[0]))(
                feature_extractor.output
            ),
        )

    def load_image(image_path):

        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = augment(img)
        img = models[config.base_model_name]["preprocess"](
            tf.cast(img, tf.float32))
        return img

    for image_id in image_ids:
        print(image_id)
        for i in range(config.num_aug_repeat if config.augment else 1):
            features_path = f"{config.features_path}/{image_id}{'_aug' + str(i) if config.augment else ''}.npz"
            if Path(features_path).exists():
                continue
            try:
                paths = list(Path(config.path).glob(f"{image_id}*"))

                image_ds = tf.data.Dataset.from_tensor_slices(
                    [str(path) for path in paths]
                )
                image_ds = image_ds.map(
                    load_image, num_parallel_calls=tf.data.AUTOTUNE
                ).batch(config.extract_batch_size)
                image_features = feature_extractor.predict(
                    image_ds, verbose=True, workers=4
                )

                np.savez_compressed(
                    features_path,
                    features=image_features,
                    image_id=image_id,
                    paths=[path.stem for path in paths],
                )

            except Exception as e:
                print(e)


def prepare_data(config):
    path = config.path
    image_ids = set([p.stem.split("_")[0]
                     for p in Path(config.path).glob(f"*.jpg")])
    if not image_ids:
        raise SystemExit(
            f"Patches not found. Make sure you have the jpg patches at {path}"
        )
    with open(config.captions_csv_path) as f:
        df = pd.read_csv(f)
    print(df.columns)
    image_id_to_caption = {}
    image_id_to_label = {}
    text_data = []
    for i in range(len(df)):
        image_id = df.iloc[i]["id"]
        if image_id in image_ids:
            caption = df.iloc[i]["text"]
            caption = "<start> " + caption.replace('"', "").strip() + " <end>"
            text_data.append(caption)
            image_id_to_caption[image_id] = caption
            image_id_to_label[image_id] = df.iloc[i]["subtype"]
    return image_id_to_caption, image_id_to_label, text_data


# Find the maximum length of any caption in the dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.noise = tf.keras.layers.GaussianNoise(stddev=0.001)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.noise(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def save_model(model: tf.keras.Model, path):
    model.save_weights(filepath=os.path.join(path, model.name + ".h5"))
    try:
        data = model.to_yaml()
        with open(os.path.join(path, model.name + ".yaml"), "w") as f:
            yaml.dump(data, f)
    except Exception as e:
        print(e)


class LogLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, optimizer=None):
        super(LogLearningRate, self).__init__()

        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):

        self._collect_learning_rate(logs)

    def _collect_learning_rate(self, logs):
        logs = logs or {}
        optimizer = self.optimizer or self.model.optimizer
        lr_schedule = getattr(optimizer, "lr", None)
        if isinstance(
            lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule
        ) or issubclass(
            type(lr_schedule), tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            logs["learning_rate"] = tf.keras.backend.get_value(
                lr_schedule(optimizer.iterations)
            )
        else:
            try:
                logs["learning_rate"] = optimizer.lr.get_value()
            except:
                pass
        return logs


class MainModel(tf.keras.Model):
    def __init__(self, encoder, decoder, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_tracker = tf.keras.metrics.Mean(name="avg_loss")

    def call(self, inputs, training=None, mask=None):
        return self.encoder(inputs)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        img_tensor, target, _ = data
        loss = 0.0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=tf.shape(target)[0])

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in tf.range(tf.shape(target)[1] - 1):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(
                    tf.expand_dims(target[:, i], 1), features, hidden
                )

                loss += self.compiled_loss(
                    target[:, i + 1], predictions, regularization_losses=self.losses
                )

        total_loss = loss / tf.cast(tf.shape(target)[1], tf.float32)

        self.loss_tracker.update_state(total_loss)

        trainable_variables = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return {m.name: m.result() for m in self.metrics + [self.loss_tracker]}

    def test_step(self, data):
        img_tensor, target, _ = data
        loss = 0.0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=tf.shape(target)[0])
        features = self.encoder(img_tensor)
        for i in tf.range(tf.shape(target)[1] - 1):
            # passing the features through the decoder
            predictions, hidden, _ = self.decoder(
                tf.expand_dims(target[:, i], 1), features, hidden
            )

            loss += self.compiled_loss(
                target[:, i + 1], predictions, regularization_losses=self.losses
            )
        total_loss = loss / tf.cast(tf.shape(target)[1], tf.float32)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Load the numpy files
def map_func(path, caption, key):
    img_tensor = np.load(path.decode("utf-8"))["features"]
    if img_tensor.ndim > 2:
        # img_tensor = tf.reduce_mean(img_tensor, axis=(1,2))
        img_tensor = np.reshape(img_tensor, (-1, img_tensor.shape[-1]))
    np.random.shuffle(img_tensor)
    img_tensor = img_tensor[:2000]

    return img_tensor, caption, key


def prepare_dataset(features, captions, keys, shuffle=True, config: Config = Config()):
    data = (features, captions, keys)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        dataset = dataset.shuffle(1000)
    # Use map to load the numpy files in parallel
    dataset = dataset.map(
        lambda item1, item2, item3: tf.numpy_function(
            map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.string]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle and batch
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds, max_length, ds_name="val", save_attention_plots=False):
        super(EvaluateCallback, self).__init__()
        self.ds = ds
        self.max_length = max_length
        self.ds_name = ds_name
        self.save_attention_plots = save_attention_plots
        Path(f"{wandb.run.dir}/attention_plots").mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):

        if logs is None:
            logs = {}
        decoder = self.model.decoder
        encoder = self.model.encoder
        tokenizer = self.model.tokenizer

        generated_captions = []
        total_bleu_score = 0

        count = 0
        for i, (img_tensor_val, caption, key) in enumerate(self.ds):

            image_id = key.numpy()[0].decode("utf-8")
            hidden = decoder.reset_state(batch_size=1)

            cap = tokenizer.sequences_to_texts(caption.numpy())
            cap = [w for w in cap[0].split() if w not in {
                "<start>", "<end>", "<pad>"}]
            features = encoder(img_tensor_val)

            attention_plot = np.zeros(
                (self.max_length, img_tensor_val.shape[1]))

            dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)
            result = []

            for i in range(self.max_length):
                predictions, hidden, attention_weights = decoder(
                    dec_input, features, hidden
                )

                attention_plot[i] = tf.reshape(
                    attention_weights, (-1,)).numpy()

                predicted_id = tf.random.categorical(predictions, 1)[
                    0][0].numpy()
                if tokenizer.index_word[predicted_id] == "<end>":
                    break
                result.append(tokenizer.index_word[predicted_id])

                dec_input = tf.expand_dims([predicted_id], 0)

            result = [w for w in result if w not in {
                "<start>", "<end>", "<pad>"}]

            bleu_score = sentence_bleu([result], cap)

            total_bleu_score += bleu_score

            if self.save_attention_plots:
                attention_plot = attention_plot[: len(result), :]
                np.savez_compressed(
                    f"{wandb.run.dir}/attention_plots/{image_id}.npz",
                    attention_plot=attention_plot,
                )
            data = dict(
                bleu_score=bleu_score,
                image_id=image_id,
                predicted=" ".join(result),
                original=" ".join(cap),
            )
            print(yaml.dump(data))
            count += 1
            generated_captions.append(data)

        if count > 0:
            logs[f"{self.ds_name}_bleu"] = total_bleu_score / count
        path = f"{wandb.run.dir}/{self.ds_name}_results_{epoch}.yaml"
        with open(path, "w") as f:
            yaml.dump(generated_captions, f)
        print(f"saved to {path}")


def run(config: Config):
    Path(config.features_path).mkdir(parents=True, exist_ok=True)

    paths = list(Path(config.path).glob("*.jpg"))
    image_ids = {path.stem.split("_")[0] for path in paths}

    image_id_to_caption, image_id_to_label, text_data = prepare_data(config)
    ids_train, ids_val, ids_test = split_train_val_ids(
        image_id_to_label=image_id_to_label
    )

    with wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        reinit=True,
        config=asdict(config),
    ) as run:
        if config.cache_features:
            extract_features_from_tiles(
                image_ids=image_ids, train_ids=ids_train, config=config
            )
        tf.keras.backend.clear_session()

        # use the ones that have extracted features
        feature_paths = list(Path(config.features_path).glob("*.npz"))
        train_ids = list({path.stem.split("_")[0] for path in feature_paths})
        image_id_to_caption = {
            k: v for k, v in image_id_to_caption.items() if k in train_ids
        }

        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            oov_token="<unk>", filters='!"#$%&()*+-/.,:;=?@[\]^_`{|}~'
        )
        tokenizer.fit_on_texts(image_id_to_caption.values())

        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        train_seqs = tokenizer.texts_to_sequences(image_id_to_caption.values())
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding="post"
        )

        image_id_to_caption_sequence = defaultdict(list)
        for img, cap in zip(image_id_to_caption.keys(), cap_vector):
            image_id_to_caption_sequence[img].append(cap)

        (
            features_path_train,
            image_ids_train,
            captions_train,
            features_path_val,
            image_ids_val,
            captions_val,
            features_path_test,
            image_ids_test,
            captions_test,
        ) = split_train_val_features(
            ids_train,
            ids_val,
            ids_test,
            image_id_to_caption_sequence=image_id_to_caption_sequence,
            feature_paths=feature_paths,
        )

        print(
            len(features_path_train),
            len(captions_train),
            len(image_ids_train),
            len(features_path_val),
            len(captions_val),
        )

        max_length = calc_max_length(train_seqs)

        train_ds = prepare_dataset(
            features_path_train, captions_train, image_ids_train, config=config
        )
        val_ds = prepare_dataset(
            features_path_val,
            captions_val,
            keys=image_ids_val,
            shuffle=False,
            config=config,
        )
        test_ds = prepare_dataset(
            features_path_test,
            captions_test,
            keys=image_ids_test,
            shuffle=False,
            config=config,
        )

        encoder = CNN_Encoder(config.embedding_dim)
        decoder = RNN_Decoder(
            config.embedding_dim, config.units, len(tokenizer.word_index)
        )
        for a in train_ds.take(1):
            encoder(a[0])

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        for name, ids in zip(
            ["train", "val", "test"], [
                image_ids_train, image_ids_val, image_ids_test]
        ):
            with Path(f"{run.dir}/{name}_ids.yaml").open("w") as f:
                yaml.dump(ids, f)

        model = MainModel(encoder, decoder, tokenizer)

        num_train_steps = len(image_ids_train)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,
            decay_steps=num_train_steps,
            decay_rate=0.98,
            staircase=True,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            loss=loss_function,
            run_eagerly=True,
        )

        callbacks = [
            EvaluateCallback(ds=val_ds, max_length=max_length),
            # tf.keras.callbacks.ModelCheckpoint(
            #     filepath=f"{wandb.run.dir or config.output_path}/checkpoint/",
            #     save_best_only=True,
            #     monitor="val_bleu",
            #     mode="max",
            #     verbose=True,
            # ),
            EvaluateCallback(
                ds=test_ds, max_length=max_length, ds_name="test"),
            tf.keras.callbacks.EarlyStopping(
                patience=10, monitor="val_bleu", mode="max"
            ),
            LogLearningRate(),
            WandbCallback(log_batch_frequency=400, save_model=False),
        ]

        try:

            model.fit(
                train_ds,
                verbose=True,
                validation_data=val_ds,
                epochs=config.epochs,
                callbacks=callbacks,
            )

        except KeyboardInterrupt:
            pass
        finally:
            save_model(encoder, wandb.run.dir)
            save_model(decoder, wandb.run.dir)


def split_train_val_ids(
    image_id_to_label,
    split=(0.7, 0.8, 1.0),
):
    random.seed(1)
    labels = set(image_id_to_label.values())
    ids_train, ids_val, ids_test = [], [], []
    for label in labels:
        image_ids = [
            image_id
            for image_id, image_label in image_id_to_label.items()
            if image_label == label
        ]
        random.shuffle(image_ids)
        n = len(image_ids)
        print(label, n)
        ids_train.extend(image_ids[: int(split[0] * n)])
        ids_val.extend(image_ids[int(split[0] * n): int(split[1] * n)])
        ids_test.extend(image_ids[int(split[1] * n): int(split[2] * n)])
    return ids_train, ids_val, ids_test


def split_train_val_features(
    ids_train,
    ids_val,
    ids_test,
    image_id_to_caption_sequence,
    feature_paths,
):
    features_path_train = []
    captions_train = []
    image_ids_train = []
    features_path_val = []
    captions_val = []
    image_ids_val = []
    features_path_test = []
    captions_test = []
    image_ids_test = []
    for path in feature_paths:
        image_id = path.stem.split("_")[0]
        path = str(path)
        if image_id in ids_train:
            features_path_train.append(path)
            captions_train.extend(image_id_to_caption_sequence[image_id])
            image_ids_train.append(image_id)
        if image_id in ids_val:
            features_path_val.append(path)
            captions_val.extend(image_id_to_caption_sequence[image_id])
            image_ids_val.append(image_id)
        if image_id in ids_test:
            features_path_test.append(path)
            captions_test.extend(image_id_to_caption_sequence[image_id])
            image_ids_test.append(image_id)

    return (
        features_path_train,
        image_ids_train,
        captions_train,
        features_path_val,
        image_ids_val,
        captions_val,
        features_path_test,
        image_ids_test,
        captions_test,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    run(args.config)
