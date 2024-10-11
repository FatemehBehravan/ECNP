from utilFiles.get_args import the_args
from models.attention import Attention
from torch.utils.data import DataLoader

args = the_args()

max_context_points = args.max_context_points

HIDDEN_SIZE = int(args.hidden_size)
MODEL_TYPE = args.model_type  # 'ANP'
ATTENTION_TYPE = args.attention_type  # 'dot_product'#'laplace'#uniform'#

save_to_dir = args.experiment_name
random_kernel_parameters = True

channels = 3
d_x, d_in, representation_size = 2, channels + 2, int(args.representation_size)
# Encoder, decoder sizes
latent_encoder_sizes = [d_in] + [HIDDEN_SIZE] * 4 + [representation_size]
determministic_encoder_sizes = [d_in] + [HIDDEN_SIZE] * 4 + [representation_size]

if args.dataset.lower() == '1d-sin' or args.dataset.lower() == 'gp' or args.dataset.lower() == 'xauusd':
    channels = 1
    d_x, d_in, representation_size = 1, channels + 1, int(args.representation_size)
    # Encoder, decoder sizes
    latent_encoder_sizes = [d_in] + [HIDDEN_SIZE] * 4 + [representation_size]
    determministic_encoder_sizes = [d_in] + [HIDDEN_SIZE] * 4 + [representation_size]

args.channels = channels



if args.use_deterministic_path and args.use_latent_path:
    decoder_sizes = [2 * representation_size + d_x] + [HIDDEN_SIZE] * 2
else:
    decoder_sizes = [representation_size + d_x] + [HIDDEN_SIZE] * 2

attention_size = [d_x] + [HIDDEN_SIZE] * 2

if MODEL_TYPE == "ANP":
    attention = Attention(representation='mlp', output_sizes=attention_size, att_type=ATTENTION_TYPE)
elif MODEL_TYPE == 'NP' or MODEL_TYPE == 'CNP':
    print("Not ANP model")
    attention = Attention(representation='identity', output_sizes=None, att_type='uniform')
else:
    raise NotImplementedError

is_custom_multi = False
# DATA GENERATOR
if 'celeba' in args.dataset.lower():
    from data.heterogeneous_data_loaders import CelebAImageLoader as ImageLoader
elif args.dataset.lower() == 'cifar10':
    from data.heterogeneous_data_loaders import Cifar10ImageLoader as ImageLoader
elif args.dataset.lower() == 'mnist':
    #print("MNIST data")
    from data.heterogeneous_data_loaders import MNISTImageLoader as ImageLoader
elif args.dataset.lower() == '1d-sin':
    from data.data_generator_1d_simple_aug2 import SinusoidCurve as generator
elif args.dataset.lower() == 'gp':
    from data.data_generator_1d_simple_aug2 import GPCurvesReader as generator
elif args.dataset.lower() == 'xauusd':
    from data.xauusd_data import NumericDataset as generator
else:
    print("Unknown dataset: ", args.dataset)
    raise NotImplementedError


if args.dataset.lower() in ['1d-sin', 'gp', 'xauusd']:
    task = "1d_regression"
elif args.dataset.lower() in ['mnist', 'celeba', 'cifar10']:
    task = "image_completion"

if task == "image_completion":
    # Training dataset
    dataset_train = DataLoader(ImageLoader("train"),batch_size=args.batch_size,shuffle=True)
    # Test dataset
    dataset_test = DataLoader(ImageLoader("test"),batch_size=1,shuffle=True)
elif task == "1d_regression":
    file_path_train = './datasets/XAUUSD/train/train_data.csv'
    file_path_test = './datasets/XAUUSD/test/test_data.csv'
    feature_columns = [ 'open']
    label_columns = ['close']
    print("generating data")
    dataset_train = generator(file_path=file_path_train,max_num_context=50, features=feature_columns, labels=label_columns, testing=False)
    dataset_test = generator(file_path=file_path_test,max_num_context=50, features=feature_columns, labels=label_columns, testing=True)
    # dataset_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # dataset_test = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("generating done")
