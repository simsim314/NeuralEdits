import os
import gc
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from Zero123PlusPipeline import Zero123PlusPipeline

class Zero123PlusSimSim314Generator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed()
        self.load_models()
        self.initialize_pipeline()
        self.freeze_parameters()
        self.setup_preprocessing()

    def set_seed(self):
        set_seed(self.config['seed'])

    def load_models(self):
        model_path = self.config['pretrained_model_name_or_path']
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler").to(self.device)

    def initialize_pipeline(self):
        self.model = Zero123PlusPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            vision_encoder=None,
            feature_extractor_clip=None,
            feature_extractor_vae=None
        ).to(self.device)

    def freeze_parameters(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.model.requires_grad_(False)
        self.unet.requires_grad_(False)

    def setup_preprocessing(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((self.config['resolution'], self.config['resolution'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def encode_image_to_latents(self, image_tensor):
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample().detach()
            return latents * self.vae.config.scaling_factor

    def generate_images(self, input_image_path):
        image_tensor = self.load_and_preprocess_image(input_image_path)
        latents = self.encode_image_to_latents(image_tensor)
        
        self.scheduler.set_timesteps(self.config['num_inference_steps'])
        latents = latents * self.scheduler.init_noise_sigma

        cond_lat = self.model.encode_condition_image(image_tensor)
        if self.config['guidance_scale'] > 1:
            negative_lat = self.model.encode_condition_image(torch.zeros_like(image_tensor))
            cond_lat = torch.cat([negative_lat, cond_lat])

        encoded = self.model.vision_encoder(image_tensor, output_hidden_states=False)
        global_embeds = encoded.image_embeds.unsqueeze(-2)

        encoder_hidden_states = self.encode_prompt("")
        ramp = global_embeds.new_tensor(self.model.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states += global_embeds * ramp

        cak = dict(cond_lat=cond_lat)
        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = None

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Generating images")):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
            
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

    def encode_prompt(self, prompt):
        text_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if hasattr(self.model, "encode_prompt"):
            return self.model.encode_prompt(prompt, self.model.device, 1, False)[0]
        else:
            return self.model._encode_prompt(prompt, self.model.device, 1, False)

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = {
        "input_image_path": "path_to_input_image",
        "output_dir": "path_to_output_dir",
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "resolution": 512,
        "seed": 42,
        "guidance_scale": 1.0,
        "num_inference_steps": 28
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    
    generator = Zero123PlusSimSim314Generator(config)
    generator.generate_images(config['input_image_path'])