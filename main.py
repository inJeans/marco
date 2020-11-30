from marco import marco

import tensorflow as tf

MARCO_TRAIN_DIR = "/scratch1/c3/data/MARCO/train-jpg"

def main():
    print("Hello")
    print("Hello again")

    train_ds = marco.load(MARCO_TRAIN_DIR,
                          as_supervised=False)
    train_ds = marco.prepare_for_training(train_ds,
                                          cache=False)

    for b in train_ds.take(1):
        # print(b[0])
        tf.keras.preprocessing.image.save_img(
            "test.png", b[1])

if __name__ == "__main__":
    main()