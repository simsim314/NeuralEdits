import os
import gc
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from accelerate import Accelerator
from Zero123PlusSimSim314Generator import Zero123PlusSimSim314Generator

class Zero123PlusSimSim314Trainer(Zero123PlusSimSim314Generator):
    def __init__(self, config):
        super().__init__(config)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.unet.requires_grad_(True)

    def train(self, input_dir, target_dir):
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.unet, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.unet,
            torch.optim.AdamW(self.unet.parameters(), lr=self.config['learning_rate']),
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['num_train_steps'])
        )

        progress_bar = tqdm(range(self.config['num_train_steps']), desc="Training")
        losses = []

        weight_dtype = torch.float32
        self.vae.to(self.accelerator.device, dtype=weight_dtype)

        for t in progress_bar:
            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file)
                target_image_path = os.path.join(target_dir, image_file)

                if not os.path.exists(target_image_path):
                    print(f"Target image not found for {image_file}")
                    continue

                image = Image.open(image_path).convert("RGB")
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.accelerator.device)

                target_image = Image.open(target_image_path).convert("RGB")
                target_image_tensor = self.preprocess(target_image).unsqueeze(0).to(self.accelerator.device)

                # Convert target image to latent space
                with torch.no_grad():
                    target_latents = self.vae.encode(target_image_tensor.to(dtype=weight_dtype)).latent_dist.sample().detach()
                    target_latents = target_latents * self.vae.config.scaling_factor

                prompt = "a photo of a "
                text_input = self.tokenizer(prompt, return_tensors="pt").to(self.accelerator.device)

                # Encode the prompt
                with torch.no_grad():
                    text_embeddings_prompt = self.text_encoder(text_input.input_ids)[0]

                # Process images
                image_1 = image_tensor
                image_2 = image_tensor
                image_1 = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
                image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
                
                # Encode condition image
                cond_lat = self.vae.encode(image_1).latent_dist.sample()
                
                # Encode vision features
                encoded = self.text_encoder.vision_model(image_2, output_hidden_states=False)
                global_embeds = encoded.image_embeds.unsqueeze(-2)
                
                # Encode prompt
                if hasattr(self.text_encoder, "encode_prompt"):
                    encoder_hidden_states = self.text_encoder.encode_prompt(prompt, self.text_encoder.device, 1, False)[0]
                else:
                    encoder_hidden_states = self.text_encoder._encode_prompt(prompt, self.text_encoder.device, 1, False)
                ramp = global_embeds.new_tensor(self.text_encoder.config.ramping_coefficients).unsqueeze(-1)
                encoder_hidden_states += global_embeds * ramp

                # Adjust for multiple images per prompt
                num_images_per_prompt = 1
                if num_images_per_prompt > 1:
                    bs_embed, *lat_shape = cond_lat.shape
                    cond_lat = cond_lat.repeat(1, num_images_per_prompt, 1, 1).view(bs_embed * num_images_per_prompt, *lat_shape)

                # Prepare cross attention kwargs
                cak = dict(cond_lat=cond_lat)
                depth_image = None  # Placeholder, set actual depth_image if using ControlNet
                if hasattr(self.unet, "controlnet"):
                    cak['control_depth'] = depth_image

                with self.accelerator.accumulate(self.unet):
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(target_latents)
                    bsz = target_latents.shape[0]
                    # Sample a random timestep for each image
                    timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device).long()
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = self.scheduler.add_noise(target_latents, noise, timestep)
                          
                    # Predict the noise using the unet
                    noise_pred = self.unet(noisy_latents, timestep, encoder_hidden_states=encoder_hidden_states, **cak).sample
                    
                    # Compute the loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    self.accelerator.backward(loss)

                    self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                losses.append(loss.detach().item())
                logs = {"lr": self.lr_scheduler.get_last_lr()[0], "avg": sum(losses[-100:]) / len(losses[-100:])}
                
                progress_bar.set_postfix(**logs)
                
                if (t + 1) % self.config['save_steps'] == 0:
                    with torch.no_grad():
                        image = self.vae.decode(1 / 0.18215 * target_latents).sample
                    
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                    images = (image * 255).round().astype("uint8")
                    pil_image = Image.fromarray(images[0])
                    
                    output_path = os.path.join(self.config['output_dir'], f"generated_image_{t+1}.png")
                    pil_image.save(output_path)

        self.cleanup()

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = {
        "input_dir": "path_to_input_dir",
        "target_dir": "path_to_target_dir",
        "output_dir": "path_to_output_dir",
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "resolution": 512,
        "seed": 42,
        "learning_rate": 1e-4,
        "num_train_steps": 1000,
        "save_steps": 100
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    
    trainer = Zero123PlusSimSim314Trainer(config)
    trainer.train(config['input_dir'], config['target_dir'])