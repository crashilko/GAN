import generator
import discriminator
from data_process import get_train_data

get_train_data()
gen = generator.make_generator_model()
generator.generate_image(gen)


