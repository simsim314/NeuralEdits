import os
import gc
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler

class Zero123PlusSimSim314Generator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed()
        self.load_models()
        self.freeze_parameters()
        self.setup_preprocessing()
        self.setup_preprocessing2()
        
    def set_seed(self):
        set_seed(self.config['seed'])

    def load_models(self):
        model_path = self.config['pretrained_model_name_or_path']
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(self.device)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="vision_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(self.device)
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    def freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vision_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

    def setup_preprocessing(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((self.config['resolution'], self.config['resolution'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def setup_preprocessing2(self):
        self.preprocess2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    def load_and_preprocess2_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess2(image).unsqueeze(0).to(self.device)

    def encode_image_to_latents(self, image_tensor):
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample().detach()
            return latents * self.vae.config.scaling_factor

    def generate_images(self, input_image_path):
        image_1 = self.load_and_preprocess_image(input_image_path)
        image_2 = self.load_and_preprocess2_image(input_image_path)
        image_1 = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        
        self.scheduler.set_timesteps(self.config['num_inference_steps'])
        latents = self.vae.encode(image_1).latent_dist.sample() * self.scheduler.init_noise_sigma

        cond_lat = self.vae.encode(image_1).latent_dist.sample()
        print(image_2.shape)
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        
        global_embeds = encoded.image_embeds.unsqueeze(-2)
        print(global_embeds.shape)
        
        text_input = self.tokenizer("", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
              text_embeddings_prompt = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
            # text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        encoder_hidden_states = text_embeddings_prompt
        ramp = global_embeds.new_tensor(self.config['ramping_coefficients']).unsqueeze(-1)

        print(encoder_hidden_states.shape, "encoder_hidden_states")
        print(global_embeds.shape, "global_embeds")
        print(ramp.shape, "ramp")
        
        # encoder_hidden_states += global_embeds * ramp

        cak = dict(cond_lat=cond_lat)
        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = None

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Generating images")):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=global_embeds*0.1).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            with torch.no_grad():
                image = self.vae.decode(1 / 0.18215 * latents).sample
            
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(images[0])
            
            output_path = os.path.join(self.config['output_dir'], f"generated_image_{i}.png")
            pil_image.save(output_path)

        self.cleanup()

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

config = {
    "input_image_path": "000008_021b957fdc234cf09da4a069cdf57a9b_0.png",
    "output_dir": "path_to_output_dir",
    "pretrained_model_name_or_path": "sudo-ai/zero123plus-v1.1",
    "resolution": 512,
    "seed": 42,
    "guidance_scale": 1.0,
    "num_inference_steps": 150,
    "ramping_coefficients": [0.1, 0.2, 0.3, 0.4]
}

os.makedirs(config['output_dir'], exist_ok=True)

generator = Zero123PlusSimSim314Generator(config)
generator.generate_images(config['input_image_path'])
