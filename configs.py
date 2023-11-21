sm_butterfly_data_config = {
    'diffusion_params' : {
        'timesteps' : 500,
        'beta1' : 1e-4,
        'beta2' : 0.02,
        # 'num_clases' : <>,
    },
    'model_params' : {
        'hidden_size' : 128,
        # 'num_classes' : <>,
        'image_size' : 128,
    },
    'training_params' : {
        'batch_size' : 16,
        'n_epoch' : 500,
        'learning_rate' : 1e-3
    },
    'ddpm' : {
        'save_dir' : './trained_models/butterflies/ddpm/'
    },
    'ddim' : {
        'save_dir' : './trained_models/butterflies/ddim/'
    },
}


sprite_data_config = {
    'diffusion_params' : {
        'timesteps' : 500,
        'beta1' : 1e-4,
        'beta2' : 0.02,
        'num_clases' : 5,
    },

    'ddpm' : {
        'save_dir' : './trained_models/ddpm/'
    },

    'ddim' : {
        'save_dir' : './trained_models/ddim/'
    },

    'model_params' : {
        'hidden_size' : 64,
        'num_classes' : 5,
        'image_size' : 16,
    },

    'training_params' : {
        'batch_size' : 100,
        'n_epoch' : 32,
        'learning_rate' : 1e-3
    },
}

