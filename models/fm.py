import torch
import torch.nn as nn
from torch.nn import functional as F
from .dit import DiT
import sys
import os
import pandas as pd
from transformers import CLIPTextModel, CLIPTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .shiftgcn.shiftgcn import Model as ShiftGCNModel




class TimeStepSampler:
    """
    Abstract class to sample timesteps for flow matching.
    """
    def sample_time(self, x_start):
        # In flow matching, time is in range [0, 1] and 1 indicates the original image; 0 is pure noise
        # this convention is *REVERSE* of diffusion
        raise NotImplementedError


class LogitNormalSampler(TimeStepSampler):
    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
        # follows https://arxiv.org/pdf/2403.03206.pdf
        # sample from a normal distribution
        # pass the output through standard logistic function, i.e., sigmoid
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start):
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        x_logistic = torch.nn.functional.sigmoid(x_normal)
        return x_logistic
    



# class MLP_FM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP_FM, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#         self.t_embedder = TimestepEmbedder(hidden_dim)
#         # Initialize timestep embedding MLP:
#         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
    
#     def forward(self, x, t):
#         B, token, dim = x.shape
#         t = self.t_embedder(t)  # B, dim
#         t_embed = t.unsqueeze(1).expand(-1, token, -1)  # B, token, dim
#         x = torch.cat((x, t_embed), dim=-1)  # B, token, dim*2
#         return self.model(x)



class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.proj_mu = nn.Linear(output_dim, output_dim)
        self.proj_log_var = nn.Linear(output_dim, output_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.backbone(x)  # n 1 768*2
        mu = self.proj_mu(x)  # n 1 768
        log_var = self.proj_log_var(x)  # n 1 768
        z0 = self.reparameterize(mu, log_var)
        return [z0, mu, log_var]

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, output_dim), 
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.backbone(x) 
        return x


class MappingNet(nn.Module):
    def __init__(self, args):
        super(MappingNet, self).__init__()
        self.args = args
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        # load pretrained models
        self.load_pretrained_models()
        # extract class text embeddings
        self.class_embeddings = self.extract_text_embeddings()
        # semantic encoder 
        self.semantic_encoder = Encoder(input_dim=args.text_dim, output_dim=args.latent_dim)  # 768*2 for sadave 
        self.motion_encoder = Encoder(input_dim=args.ske_dim, output_dim=args.latent_dim)  # 256 for sadave
        # decoder
        self.motion_decoder = Decoder(input_dim=args.latent_dim, output_dim=args.ske_dim)  # 768-> 256
        self.semantic_decoder = Decoder(input_dim=args.latent_dim, output_dim=args.text_dim)  # 768 -> 1536
        # load mapping network - DiT
        self.time_step_sampler = LogitNormalSampler(0, 1)
        self.velocity_predictor = DiT(
            in_channels=args.latent_dim,  # 768
            hidden_size=args.latent_dim,   # 768
            depth=args.dit_layers,  # 12
            num_heads=args.num_heads,   # 12
            mlp_ratio=args.mlp_ratio,  # 4.0
        )
        # self.velocity_predictor = MLP_FM(input_dim=args.latent_dim*2, hidden_dim=args.latent_dim, output_dim=args.latent_dim)  # 768*2 -> 768
        
        
    def load_pretrained_models(self):
        # load shiftgcn model
        if self.args.shiftgcn_checkpoint_path is not None:
            self.shiftgcn = ShiftGCNModel(num_class=self.args.num_class, num_point=self.args.num_point, num_person=self.args.num_person, graph=self.args.graph, graph_args=self.args.graph_args).to('cuda')
            self.shiftgcn.load_state_dict(torch.load(self.args.shiftgcn_checkpoint_path, map_location='cpu'))
            self.shiftgcn.requires_grad_(False)
        # load clip text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.clip_checkpoint_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.clip_checkpoint_path)
        self.text_encoder = self.text_encoder.to('cuda')
        self.text_encoder.requires_grad_(False)
    
    
    def extract_text_embeddings(self):
        dataset_name = self.args.task_name.split('_')[0]  # e.g., 'ntu60'
        # deal with prompts
        def deal_prompts(prompts):
            text_embed = []
            for prompt in prompts:
                text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.args.max_length, truncation=True, return_tensors="pt").to('cuda')  
                embeds = self.text_encoder(text_inputs.input_ids)
                prompt_embeds = embeds['last_hidden_state']
                pooled_prompt_embeds = embeds['pooler_output']
                text_embed.append(torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone())  
            return text_embed
        # type1: csv file
        csv_file = os.path.join(self.args.semantic_path, '{}.csv'.format(dataset_name)) 
        csv_prompts = pd.read_csv(csv_file)['label'].values.tolist() 
        text_embed_csv = deal_prompts(csv_prompts)
        # type2: txt file
        txt_file = os.path.join(self.args.semantic_path, '{}_llm.txt'.format(dataset_name))
        with open(txt_file, 'r', encoding='utf-8') as file:
            content_list = file.readlines()
        txt_prompts = [line.strip() for line in content_list]
        text_embed_txt = deal_prompts(txt_prompts)
        # concatenate type 1 & type 2
        self.class_embeddings = torch.cat((torch.cat(text_embed_csv, dim=0), torch.cat(text_embed_txt, dim=0)), dim=-1)  # [60, token, 768*2]
        return self.class_embeddings # [60, token, 768*2]
        

    def psi(self, t, x, x1):
        assert (
            t.shape[0] == x.shape[0]
        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
        assert t.ndim == 1
        t = self.expand_t(t, x)
        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1
    
    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)
    
    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
        assert x.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x + x1
    
    def prepare_embedding(self, skeleton_data, seen_classes, label_idx, attuned_semantics):
        # skeleton -- prepare data
        if not self.args.use_features:
            skeleton_embedding = self.shiftgcn(skeleton_data) 
            B, c_motion, t, v = skeleton_embedding.size()
            skeleton_embedding = skeleton_embedding.mean(3).mean(2)   # b 256
        else:
            skeleton_embedding = skeleton_data  # b 256
            B, _ = skeleton_embedding.size()  # b 256

        skeleton_embedding = skeleton_embedding.unsqueeze(1).expand(-1, self.args.max_length+1, -1)  # b token 256
        # semantics -- prepare data
        semantics_seen = attuned_semantics[seen_classes]  # seen token 768*2
        semantics_embedding = torch.cat([semantics_seen[i,:,:].unsqueeze(0) for i in label_idx], dim=0)  # b token 768*2
        return skeleton_embedding, semantics_embedding  # b token 256, b token 768*2
    
    
    def forward_phase1(self, skeleton_data, attuned_semantics, seen_classes, label_idx):
        # freeze Flow Matching model
        for p in self.velocity_predictor.parameters():
            p.requires_grad = False
        for p in self.semantic_encoder.parameters():
            p.requires_grad = True
        for p in self.motion_encoder.parameters():
            p.requires_grad = True
        for p in self.motion_decoder.parameters():
            p.requires_grad = True
        for p in self.semantic_decoder.parameters():
            p.requires_grad = True
        # prepare skeleton & semantics embedding
        skeleton_embedding, semantics_embedding = self.prepare_embedding(skeleton_data, seen_classes, label_idx, attuned_semantics)  # b token 256, b token 768*2
        B, _, _ = skeleton_embedding.size()  # b token 256
        # encoders
        z0_text, mu_text, log_var_text = self.semantic_encoder(semantics_embedding)  # b token 768
        z1_ske, mu_ske, log_var_ske = self.motion_encoder(skeleton_embedding)  # b token 768
        # mu & logvar
        loss_mu = F.mse_loss(mu_text, mu_ske, reduction='mean')  
        loss_log_var = F.mse_loss(log_var_text, log_var_ske, reduction='mean') 
        # MSE reconstruction loss
        recon_loss_s2s = F.mse_loss(self.motion_decoder(z1_ske), skeleton_embedding, reduction='mean')  # MSE loss for skeleton
        recon_loss_t2t = F.mse_loss(self.semantic_decoder(z0_text), semantics_embedding, reduction='mean')  # MSE loss for text
        recon_loss_s2t = F.mse_loss(self.semantic_decoder(z1_ske), semantics_embedding, reduction='mean')  # MSE loss for skeleton to text
        recon_loss_t2s = F.mse_loss(self.motion_decoder(z0_text), skeleton_embedding, reduction='mean')  # MSE loss for text to skeleton
        loss_recon = (recon_loss_s2s + recon_loss_t2t + recon_loss_s2t + recon_loss_t2s) / 4  # average MSE loss
        # total loss
        total_loss = loss_recon + self.args.lambda_align * (loss_mu + loss_log_var)
        
        return {
            'loss': total_loss,
            'loss_mu': loss_mu,
            'loss_log_var': loss_log_var,
            'loss_recon': loss_recon,
        }         
    
    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx, label_idx_contrastive, seen_classes, attuned_semantics):
        # freeze Flow Matching model
        for p in self.velocity_predictor.parameters():
            p.requires_grad = True
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False
        for p in self.motion_encoder.parameters():
            p.requires_grad = False
        for p in self.motion_decoder.parameters():
            p.requires_grad = False
        for p in self.semantic_decoder.parameters():
            p.requires_grad = False
        # prepare skeleton & semantics embedding
        skeleton_embedding, semantics_embedding = self.prepare_embedding(skeleton_data, seen_classes, label_idx, attuned_semantics)  # b token 256, b token 768*2
        skeleton_embedding_contrastive, semantics_embedding_contrastive = self.prepare_embedding(skeleton_data_contraastive, seen_classes, label_idx_contrastive, attuned_semantics)  # b token 256, b token 768*2
        # encoders
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)  # b token 768
        z1_ske, _, _ = self.motion_encoder(skeleton_embedding)  # b token 768
        # encoders - contrastive
        z0_text_contrastive, _, _ = self.semantic_encoder(semantics_embedding_contrastive)  # b token 768
        z1_ske_contrastive, _, _ = self.motion_encoder(skeleton_embedding_contrastive)  # b token 768
        # flow matching  -- text to skeleton
        time_steps = self.time_step_sampler.sample_time(z0_text)
        z_time = self.psi(time_steps, z0_text, z1_ske)
        target_velocity = self.Dt_psi(time_steps, z0_text, z1_ske)
        pred_velocity = self.velocity_predictor(z_time, time_steps)
        flow_matching_loss = F.mse_loss(pred_velocity, target_velocity, reduction='mean')  # flow matching loss
        # flow matching -- contrastive learning
        target_velocity_contrastive = self.Dt_psi(time_steps, z0_text_contrastive, z1_ske_contrastive)
        flow_matching_loss_contrastive = F.mse_loss(pred_velocity, target_velocity_contrastive, reduction='mean')  # flow matching loss
        # total loss
        total_loss = flow_matching_loss - self.args.lambda_cfm * flow_matching_loss_contrastive # total loss

        return {
            'loss': total_loss,
            'loss_flow_matching': flow_matching_loss,
            'loss_flow_matching_contrastive': flow_matching_loss_contrastive,
        }  

    
    @torch.no_grad()
    def predict_zsl(self, skeleton_data, unseen_classes, attuned_semantics, label):
        # prepare skeleton embedding
        if not self.args.use_features:
            skeleton_embedding = self.shiftgcn(skeleton_data) 
            B, _, _, _ = skeleton_embedding.size()
            skeleton_embedding = skeleton_embedding.mean(3).mean(2)   # b 256
        else:
            skeleton_embedding = skeleton_data
            B, _ = skeleton_embedding.size()  # b 256
        
        skeleton_embedding = skeleton_embedding.unsqueeze(1).expand(-1, self.args.max_length+1, -1)  # b token 256
        z1_ske, _, _ = self.motion_encoder(skeleton_embedding)  # b token 768
        # prepare semantics embedding
        semantics_embedding = attuned_semantics[unseen_classes]  # unseen token 768*2
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)  # unseen token 768
        # timestep
        timestep = torch.ones(B, device=skeleton_embedding.device) * self.args.step_ratio  # [B]
        mse_error_list = []
        for i, _ in enumerate(unseen_classes):
            # flow matching
            z0_text_category = z0_text[i].unsqueeze(0).expand(B, -1, -1)  # [B, token, 768]
            z_time = self.psi(timestep, z0_text_category, z1_ske)  # [B, token, 768]  # z_time
            pred_velocity = self.velocity_predictor(z_time, timestep)  # [B, token, 768]  
            target_velocity = self.Dt_psi(timestep, z0_text_category, z1_ske)  # [B, token, 768]  
            mse_error = F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(2).mean(1) # B
            mse_error_list.append(mse_error.view(B, 1))
        mse_error_list = torch.cat(mse_error_list, dim=1)  # B, unseen_classes
        predictions_idx = torch.argmin(mse_error_list, dim=1)  # B, 
        return {
            'predictions_idx': predictions_idx,  # B, 
        }
    
    @torch.no_grad()
    def predict_gzsl(self, skeleton_data, seen_classes, unseen_classes, attuned_semantics):
        # prepare skeleton embedding
        if not self.args.use_features:
            skeleton_embedding = self.shiftgcn(skeleton_data) 
            B, _, _, _ = skeleton_embedding.size()
            skeleton_embedding = skeleton_embedding.mean(3).mean(2)   # b 256
        else:
            skeleton_embedding = skeleton_data
            B, _ = skeleton_embedding.size()  # b 256
        
        skeleton_embedding = skeleton_embedding.unsqueeze(1).expand(-1, self.args.max_length+1, -1)  # b token 256
        z1_ske, _, _ = self.motion_encoder(skeleton_embedding)  # b_unseen token 768
        # prepare semantics embedding
        semantics_embedding = attuned_semantics  # seen+unseen token 768*2
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)  # unseen token 768
        # timestep
        timestep = torch.ones(B, device=skeleton_embedding.device) * self.args.step_ratio  # [B]
        mse_error_list = []
        for i in range(len(seen_classes)+len(unseen_classes)):
            # flow matching
            z0_text_category = z0_text[i].unsqueeze(0).expand(B, -1, -1)
            z_time = self.psi(timestep, z0_text_category, z1_ske)  # [B, token, 768]  # z_time
            pred_velocity = self.velocity_predictor(z_time, timestep)  # [B, token, 768]  
            target_velocity = self.Dt_psi(timestep, z0_text_category, z1_ske)  # [B, token, 768]  
            mse_error = F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(2).mean(1) # B
            mse_error_list.append(mse_error.view(B, 1))
        mse_error_list = torch.cat(mse_error_list, dim=1)  # B, unseen_classes + seen_classes
        # determine the skeleton sample is from seen domain or unseen domain
        mse_error_seen_min = torch.min(mse_error_list[:, seen_classes], dim=1).values   # B
        mse_error_unseen_min = torch.min(mse_error_list[:, unseen_classes], dim=1).values  # B
        ratio = mse_error_seen_min / mse_error_unseen_min
        row_seen = (ratio > self.args.calibration_factor).unsqueeze(1)  # B, 1
        row_unseen = ~row_seen
        col_seen = torch.zeros(1, len(seen_classes) + len(unseen_classes), dtype=torch.bool, device=skeleton_embedding.device)
        col_seen[:, seen_classes] = True
        col_unseen = torch.zeros(1, len(seen_classes) + len(unseen_classes), dtype=torch.bool, device=skeleton_embedding.device)
        col_unseen[:, unseen_classes] = True
        mse_error_list[row_seen & col_seen] = 1e6  # large penalty to seen categories
        mse_error_list[row_unseen & col_unseen] = 1e6  # large penalty to unseen categories
        predictions_idx = torch.argmin(mse_error_list, dim=1)  # B,
        return {
            'predictions_idx': predictions_idx,  # B, 
        }























