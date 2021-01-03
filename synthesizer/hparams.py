from tensorflow.contrib.training import HParams

# Default hyperparameters
hparams = HParams(
    cleaners="english_cleaners",

    tacotron_gpu_start_idx=0,  # idx of the first GPU to be used for Tacotron training.
    tacotron_num_gpus=1,  # Determines the number of gpus in use for Tacotron training.
    split_on_cpu=True,

    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality

    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    clip_mels_length=True,

    max_mel_frames=900,

    use_lws=False,

    silence_threshold=2,  # silence threshold used for sound trimming for wavenet preprocessing
    
    # Mel spectrogram  
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    

    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    

    signal_normalization=True,

    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,

    max_abs_value=4.,

    normalize_for_wavenet=True,

    clip_for_wavenet=True,

    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,

    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,

    griffin_lim_iters=60,

    

    outputs_per_step=2, # Was 1

    stop_at_any=True,

    
    embedding_dim=512,  # dimension of embedding space (these are NOT the speaker embeddings)
    
    # Encoder parameters
    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=512,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=256,  # number of lstm units for each direction (forward and backward)
    

    smoothing=False,  # Whether to smooth the attention normalization function
    attention_dim=128,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31,),  # kernel size of attention convolution
    cumulative_weights=True,

    
    # Decoder
    prenet_layers=[256, 256],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=1024,  # number of decoder lstm units on each layer
    max_iters=2000,
    # Max decoder steps during inference (Just for safety from infinite loop cases)
    
    # Residual postnet
    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=512,  # number of postnet convolution filters for each layer
    
    # CBHG mel->linear postnet
    cbhg_kernels=8,
    # All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act
    #  as "K-grams"
    cbhg_conv_channels=128,  # Channels of the convolution bank
    cbhg_pool_size=2,  # pooling size of the CBHG
    cbhg_projection=256,
    # projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
    cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
    cbhg_highwaynet_layers=4,  # Number of HighwayNet layers
    cbhg_highway_units=128,  # Number of units used in HighwayNet fully connected layers
    cbhg_rnn_units=128,
    # Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in 
    # shape
    
    # Loss params
    mask_encoder=True,
    # whether to mask encoder padding while computing attention. Set to True for better prosody 
    # but slower convergence.
    mask_decoder=False,
    # Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not
    #  be weighted, else recommended pos_weight = 20)
    cross_entropy_pos_weight=20,
    # Use class weights to reduce the stop token classes imbalance (by adding more penalty on 
    # False Negatives (FN)) (1 = disabled)
    predict_linear=False,
    # Whether to add a post-processing network to the Tacotron to predict linear spectrograms (
	# True mode Not tested!!)
    ###########################################################################################################################################

    # Tacotron Training
    # Reproduction seeds
    tacotron_random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    tacotron_data_random_state=1234,  # random state for train test split repeatability
    
    # performance parameters
    tacotron_swap_with_cpu=False,
    # Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause 
    # major slowdowns! Only use when critical!)
    
    # train/test split ratios, mini-batches sizes
    tacotron_batch_size=36,  # number of training samples on each training steps (was 32)

    tacotron_synthesis_batch_size=128,
    # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN"T TRAIN TACOTRON WITH "mask_encoder=True"!!
    tacotron_test_size=0.05,

    tacotron_test_batches=None,  # number of test batches.
    

    tacotron_decay_learning_rate=True,
    # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=50000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate
    
    # Optimization parameters
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter
    
    # Regularization parameters
    tacotron_reg_weight=1e-7,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=False,
    # Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is
    #  high and biasing the model)
    tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet
    tacotron_clip_gradients=True,  # whether to clip gradients
    
    # Evaluation parameters
    natural_eval=False,

    tacotron_teacher_forcing_mode="constant",

    tacotron_teacher_forcing_ratio=1.,

    tacotron_teacher_forcing_init_ratio=1.,
    # initial teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_final_ratio=0.,
    # final teacher forcing ratio. Relevant if mode="scheduled"
    tacotron_teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode="scheduled"
    tacotron_teacher_forcing_decay_alpha=0.,

 
    # Tacotron-2 integration parameters
    train_with_GTA=False,

    sentences=[
        # From July 8, 2017 New York Times:
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There\"s a way to measure the acute emotional intelligence that has never gone out of "
		"style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "The Senate\"s bill to repeal and replace the Affordable Care Act is now imperiled.",
        # From Google"s Tacotron example page:
        "Generative adversarial network or variational auto-encoder.",
        "Basilar membrane and otolaryngology are not auto-correlations.",
        "He has read the whole thing.",
        "He reads books.",
        "He thought it was time to present the present.",
        "Thisss isrealy awhsome.",
        "Punctuation sensitivity, is working.",
        "Punctuation sensitivity is working.",
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",
        # From The web (random long utterance)
        "Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
        This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
        the adopted architecture is able to perform this task with wild success.",
        "Thank you so much for your support!",
    ],
    
    
    ### SV2TTS ###
    speaker_embedding_size=256,
    silence_min_duration_split=0.4, # Duration in seconds of a silence for an utterance to be split
    utterance_min_duration=1.6,     # Duration in seconds below which utterances are discarded
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
